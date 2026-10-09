#!/usr/bin/env python
"""Tour the room and the hall, map everything, and come back to the exact starting pose.

roomrun.py picks the single most open bearing each leg. That is fine for one room: it walks into
open space and stops when nothing is open. It cannot tour a room AND a hall, because once the hall
is entered the most open bearing keeps pointing further down it, and once it is walked the most
open bearing points back the way it came. It has no notion of "somewhere I have not looked yet".

This one is frontier-driven instead. A frontier is a known-free cell touching unknown space - the
boundary of what has been seen. Driving to frontiers is what makes the coverage complete: the room
gets finished before the hall is entered only if the room's frontiers are nearer, and the run ends
when no frontier remains, not when the robot happens to face a wall.

Two things here are deliberately not roomrun.py's:

  * nav.corridor_clear returns (ok, reason, distance). roomrun.py tests it as `if not
    nav.corridor_clear(...)`, and a non-empty tuple is always truthy, so that check never fires.
    Unpacked properly here.

  * Every leg is re-planned from the live map. A frontier chosen two legs ago may have been mapped
    by the laser in the meantime without ever being driven to, and is then no longer a frontier.
"""
import json, math, time, sys, os
import numpy as np
import rospy
from goto import Nav

# --- budget -------------------------------------------------------------------------------
MAX_LEGS      = 40          # a room plus a hall, not a building
MAX_MINUTES   = 12.0        # wall-clock ceiling for the exploring phase
MAX_TRAVEL_M  = 60.0        # total path length ceiling

# --- geometry -----------------------------------------------------------------------------
NEAR_M        = 1.50        # ignore frontiers this close: nibbling them wastes the budget
FAR_M         = 14.0        # far enough to commit to the hall in one leg
STANDOFF_M    = 0.45        # stop this far short of the frontier, not on top of it
MIN_CLUSTER   = 25          # cells; only openings big enough to be a doorway or a hall
LEG_TMO       = 75          # s per leg
RETURN_TMO    = 180         # s for the final run home

# --- failure handling ---------------------------------------------------------------------
MAX_CONSEC_FAIL = 3         # give up exploring after this many legs in a row fail
STUCK_M         = 0.08      # a leg that moves less than this counts as a failure

FREE_MAX = 40               # grid value below this is free enough to stand on


def frontier_clusters(nav):
    """Known-free cells that touch unknown space, grouped into connected clusters.

    Returns [(world_x, world_y, size), ...] sorted by size, biggest first.
    """
    g = nav.grid
    if g is None:
        return []
    free = (g >= 0) & (g < FREE_MAX)
    unknown = (g < 0)
    # a free cell touching unknown in any of the 4 directions
    touch = np.zeros_like(free)
    touch[:-1, :] |= unknown[1:, :]
    touch[1:, :] |= unknown[:-1, :]
    touch[:, :-1] |= unknown[:, 1:]
    touch[:, 1:] |= unknown[:, :-1]
    front = free & touch
    ys, xs = np.nonzero(front)
    if len(xs) == 0:
        return []

    # Cluster by 8-connectivity with a flood fill over the sparse set. The frontier is a thin
    # boundary curve, so the cell count stays small and an explicit stack is fast enough.
    pts = set(zip(xs.tolist(), ys.tolist()))
    info = nav.map.info
    out = []
    while pts:
        seed = pts.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cx, cy = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    p = (cx + dx, cy + dy)
                    if p in pts:
                        pts.discard(p)
                        stack.append(p)
                        comp.append(p)
        if len(comp) < MIN_CLUSTER:
            continue
        mx = sum(p[0] for p in comp) / float(len(comp))
        my = sum(p[1] for p in comp) / float(len(comp))
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        out.append((wx, wy, len(comp)))
    out.sort(key=lambda c: -c[2])
    return out


def pick_goal(nav, x, y, clusters, visited):
    """Nearest worthwhile frontier, backed off to a spot the robot can actually stand on.

    Yields candidates best-first so the caller can fall through to the next one when move_base
    refuses. Scoring is SIZE-dominated with a mild distance penalty. Distance-dominated scoring
    was tried first and it live-locked: standing in a mapped room, the nearest frontier is always
    a few cells of shadow behind a chair leg, so the robot spent every leg doing a 160 deg turn to
    gain 0.09 m instead of leaving for the hall. Size first fixes that - a big opening is a place
    worth going - and the distance term only breaks ties between comparable openings. The cap at
    400 stops one enormous frontier from outvoting a nearer one forever.
    """
    cands = []
    for (fx, fy, n) in clusters:
        d = math.hypot(fx - x, fy - y)
        if d < NEAR_M or d > FAR_M:
            continue
        if any(math.hypot(fx - vx, fy - vy) < 0.7 for (vx, vy) in visited):
            continue
        cands.append((-min(n, 400) + 8.0 * d, fx, fy, n, d))
    cands.sort()

    for _, fx, fy, n, d in cands:
        a = math.atan2(fy - y, fx - x)
        # back off along the approach bearing so the goal is inside known space
        for back in (STANDOFF_M, STANDOFF_M + 0.25, STANDOFF_M + 0.5):
            gx, gy = fx - back * math.cos(a), fy - back * math.sin(a)
            ok, why = nav.goal_is_standable(gx, gy)
            if not ok:
                continue
            if nav.cost is not None:
                ok2, _ = nav.goal_is_reachable_cell(gx, gy)
                if not ok2:
                    continue
            yield (gx, gy, a, fx, fy, n, d)
            break


def known_m2(nav):
    if nav.grid is None:
        return 0.0
    return int((nav.grid >= 0).sum()) * (nav.map.info.resolution ** 2)


def tighten(xy, yaw):
    try:
        import dynamic_reconfigure.client as dc
        for ns in ('/move_base/TebLocalPlannerROS', '/move_base/DWAPlannerROS'):
            try:
                c = dc.Client(ns, timeout=3.0)
                c.update_configuration({'xy_goal_tolerance': xy, 'yaw_goal_tolerance': yaw})
                print('  tolerance set on %s -> %.2f m / %.2f rad' % (ns, xy, yaw))
                return True
            except Exception:
                pass
    except Exception:
        pass
    print('  (could not set tolerance; using the planner default)')
    return False


def main():
    nav = Nav()
    time.sleep(2.5)

    sx, sy, syaw = nav.pose()
    json.dump({'x': sx, 'y': sy, 'yaw_deg': math.degrees(syaw)},
              open('/home/jetson/START_POSE.json', 'w'))
    m0 = known_m2(nav)
    print('START POSE  (%.3f, %.3f) yaw %.1f deg' % (sx, sy, math.degrees(syaw)))
    print('map known   %.1f m2' % m0)
    print('budget      %d legs / %.0f min / %.0f m' % (MAX_LEGS, MAX_MINUTES, MAX_TRAVEL_M))

    t_start = time.time()
    travelled = 0.0
    legs = 0
    consec_fail = 0
    visited = []

    for leg in range(MAX_LEGS):
        if time.time() - t_start > MAX_MINUTES * 60:
            print('\ntime budget reached'); break
        if travelled > MAX_TRAVEL_M:
            print('\ndistance budget reached'); break
        if consec_fail >= MAX_CONSEC_FAIL:
            print('\n%d legs failed in a row, stopping exploration' % consec_fail); break

        x, y, yaw = nav.pose()
        clusters = frontier_clusters(nav)
        print('\n--- leg %d ---  at (%.2f, %.2f)  %d frontier clusters  %.1f m2 known  %.0f s elapsed'
              % (leg + 1, x, y, len(clusters), known_m2(nav), time.time() - t_start))
        if not clusters:
            print('  no frontiers left: everything reachable has been seen'); break

        sent = False
        tried = 0
        for (gx, gy, a, fx, fy, n, d) in pick_goal(nav, x, y, clusters, visited):
            tried += 1
            if tried > 4:
                break
            ok, why, _ = nav.corridor_clear(x, y, gx, gy)
            note = 'direct' if ok else 'planner must route around (%s)' % why
            print('  -> frontier %d cells at %.2f m, goal (%.2f, %.2f), %s' % (n, d, gx, gy, note))
            if not nav.send(gx, gy, a):
                continue
            if nav.watch(gx, gy, timeout=LEG_TMO):
                nx, ny, _ = nav.pose()
                moved = math.hypot(nx - x, ny - y)
                travelled += moved
                visited.append((fx, fy))
                legs += 1
                sent = True
                consec_fail = 0 if moved >= STUCK_M else consec_fail + 1
                if moved < STUCK_M:
                    print('  arrived but barely moved (%.2f m)' % moved)
            else:
                nav.stop()
                consec_fail += 1
                visited.append((fx, fy))   # do not retry a frontier move_base refused
                print('  leg failed (%d in a row)' % consec_fail)
                sent = True
            break
        if not sent:
            print('  no standable goal near any frontier')
            consec_fail += 1
        time.sleep(1.5)   # let gmapping integrate the new scans before choosing again

    # ---- home, always, whatever happened above --------------------------------------------
    print('\n=== returning to start ===')
    tighten(0.05, 0.10)
    home = False
    for attempt in (1, 2):
        if nav.send(sx, sy, syaw) and nav.watch(sx, sy, timeout=RETURN_TMO):
            home = True
            break
        print('  return attempt %d did not complete' % attempt)
        nav.stop()
        time.sleep(2.0)

    time.sleep(1.5)
    fx_, fy_, fyaw = nav.pose()
    err = math.hypot(fx_ - sx, fy_ - sy)
    dyaw = math.degrees(math.atan2(math.sin(fyaw - syaw), math.cos(fyaw - syaw)))
    m1 = known_m2(nav)

    print('\n=== RESULT ===')
    print('legs driven        : %d' % legs)
    print('distance travelled : %.2f m' % travelled)
    print('elapsed            : %.1f min' % ((time.time() - t_start) / 60.0))
    print('map known          : %.1f -> %.1f m2  (+%.1f)' % (m0, m1, m1 - m0))
    print('returned home      : %s' % ('yes' if home else 'NO - move_base did not reach it'))
    print('start was          : (%.3f, %.3f) yaw %.1f' % (sx, sy, math.degrees(syaw)))
    print('ended at           : (%.3f, %.3f) yaw %.1f' % (fx_, fy_, math.degrees(fyaw)))
    print('POSITION ERROR     : %.3f m' % err)
    print('HEADING ERROR      : %.1f deg' % dyaw)
    nav.stop()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\ninterrupted - stopping the robot')
        try:
            Nav().stop()
        except Exception:
            pass
        sys.exit(1)
