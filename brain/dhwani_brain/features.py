# -*- coding: utf-8 -*-
"""Everything a policy may look at, computed from ONE obs dict plus the brain's own memory.

build_context() turns obs into: the planning grid, geodesic fields from the robot / home /
last-link point, frontiers, person standoff points, the energy/time/distance model, and a list of
candidate actions with feasibility and a fixed-length feature vector each. Both policies (rules
and learned) choose among exactly these candidates, so the learned one can never invent a goal.

Every feature is listed in CAND_FEATURES / GLOBAL_FEATURES with its normalisation; FEATURES.md is
generated from those lists (python -m dhwani_brain.features > FEATURES.md) so the doc cannot drift
from the code.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

from .compat import clock, fnum, wrap, INF
from .mapview import MapView, box_count
from .geodesic import Graph, distances, descend
from .frontiers import find_frontiers, search_goals

KINDS = ("EXPLORE", "VERIFY", "RETURN_HOME", "RETURN_TO_LINK", "HOLD")
TASK_ORDER = ("SEARCH_SURVIVORS", "MAP_AREA", "GOTO_WAYPOINT", "RETURN_HOME", "HALT")
PERSON_ORDER = ("REPORT_AND_CONTINUE", "REPORT_AND_RETURN", "STAY")
LINK_ORDER = ("CONTINUE_THEN_RETURN", "RETURN_NOW", "CONTINUE", "HOLD")
BRANCH_ORDER = ("ANY", "LEFT", "RIGHT", "STRAIGHT")

# (name, normalisation / meaning) - the order IS the feature vector layout
CAND_FEATURES = [
    ("is_explore", "1 if the candidate is an exploration goal (lidar frontier or camera search)"),
    ("is_verify", "1 if it is a standoff point to verify an unconfirmed sighting"),
    ("is_home", "1 if it is RETURN_HOME"),
    ("is_link", "1 if it is RETURN_TO_LINK (last pose where the link worked)"),
    ("is_hold", "1 if it is HOLD (stay where you are)"),
    ("d_robot", "geodesic robot->goal, m / 50, clipped to [0, 3]"),
    ("d_home", "geodesic goal->home, m / 50, clipped to [0, 3]; 3 if home unreachable"),
    ("d_euclid", "straight-line robot->goal, m / 50, clipped to [0, 3]"),
    ("detour", "geodesic / max(straight line, 1 m), / 3, clipped to [0, 1]"),
    ("sin_bear", "sin of goal bearing relative to the robot heading"),
    ("cos_bear", "cos of goal bearing relative to the robot heading"),
    ("sin_bear0", "sin of goal bearing relative to the heading at mission start"),
    ("cos_bear0", "cos of goal bearing relative to the heading at mission start"),
    ("branch_match", "1 if first_branch is still pending and the goal lies in its sector"),
    ("is_search", "1 if it is a camera-search goal (known floor the camera has not been near)"),
    ("size", "frontier length m / 5, or unseen search area m2 / 10; clipped to [0, 2]; else 0"),
    ("unknown_frac", "unknown cells / all cells in a 6 m x 6 m box around the goal"),
    ("openness", "passable cells / all cells in a 3.4 m x 3.4 m box around the goal"),
    ("bat_margin", "(battery - energy to goal - reserve from goal - floor) / 100, [-1, 1]"),
    ("time_margin", "(time limit - time needed incl. return) / limit, [-1, 1]; 1 if no limit"),
    ("dist_margin", "(distance limit - odometry needed incl. return) / limit, [-1, 1]; 1 if none"),
    ("utility", "log1p(rules frontier utility x 10) / 3; 0 for non-frontiers"),
    ("rules_pick", "1 if the rules policy chose this candidate"),
    ("is_prev_goal", "1 if within 1.5 m of the previous decision's goal"),
    ("d_home_delta", "(d_home(goal) - d_home(robot)) / 50, [-2, 2]: >0 means deeper"),
    ("beyond_link", "(d_home(goal) - d_home(last link point)) / 50, [-2, 2]; 0 if none"),
    ("person_near", "exp(-d / 3 m), d = goal to nearest unconfirmed sighting; 0 if none"),
    ("visited", "fraction of trail points within 2 m of the goal"),
    ("blocked_near", "exp(-d / 3 m), d = goal to nearest blocked region edge; 0 if none"),
    ("rank", "rank by rules utility / K among frontiers (0 = best); 1 for non-frontiers"),
    ("hits", "sighting hits / 5, clipped to [0, 1] (VERIFY only)"),
]
GLOBAL_FEATURES = [
    ("battery", "battery_pct / 100"),
    ("reserve_margin", "(battery - floor - reserve to get home now) / 100, [-1, 1]"),
    ("link_ok", "1 if the dashboard heartbeat is live"),
    ("since_contact", "since_contact_s / 600, clipped to [0, 1]"),
    ("elapsed", "elapsed_s / 3600, clipped to [0, 1.5]"),
    ("time_frac", "elapsed / max_duration, [0, 1.5]; 0 if no limit"),
    ("has_time_limit", "1 if max_duration_min is set"),
    ("odo", "odo_m / 1000, clipped to [0, 1.5]"),
    ("dist_frac", "odo / max_distance, [0, 1.5]; 0 if no limit"),
    ("has_dist_limit", "1 if max_distance_m is set"),
    ("d_home_now", "geodesic robot->home, m / 100, clipped to [0, 2]; 2 if unreachable"),
    ("home_reachable", "1 if a known route home exists"),
    ("found", "confirmed persons / 3, clipped to [0, 1]"),
    ("unreported", "confirmed but not yet reported / 3, clipped to [0, 1]"),
    ("unconfirmed", "open (not written-off) unconfirmed sightings / 3, clipped to [0, 1]"),
    ("known_free", "known free area, m2 / 2000, clipped to [0, 1.5]"),
    ("cam_covered", "fraction of reachable known floor within cam_cover_m of the trail"),
    ("n_search", "camera-search goals / 10, clipped to [0, 1.5]"),
    ("n_frontiers", "frontier clusters / 10, clipped to [0, 1.5]"),
    ("frontier_len", "total frontier length, m / 20, clipped to [0, 1.5]"),
    ("n_blocked", "blocked regions / 3, clipped to [0, 1]"),
    ("link_known", "1 if a last-link pose exists"),
    ("d_link", "geodesic robot->last-link point, m / 50, [0, 2]; 0 if none or link ok"),
    ("branch_pending", "1 if first_branch is not ANY and not yet taken"),
    ("min_battery", "mission min_battery_pct / 100; 0 if none"),
    ("speed_slow", "1 if mission speed is SLOW"),
    ("return_when_done", "mission flag"),
    ("report_person", "mission flag"),
    ("approach_person", "mission flag"),
] + [("task_" + t.lower(), "mission task one-hot") for t in TASK_ORDER] \
  + [("opf_" + t.lower(), "on_person_found one-hot") for t in PERSON_ORDER] \
  + [("oll_" + t.lower(), "on_link_loss one-hot") for t in LINK_ORDER] \
  + [("fb_" + t.lower(), "first_branch one-hot") for t in BRANCH_ORDER]

NC = len(CAND_FEATURES)
NG = len(GLOBAL_FEATURES)
CI = dict((n, i) for i, (n, _) in enumerate(CAND_FEATURES))
GI = dict((n, i) for i, (n, _) in enumerate(GLOBAL_FEATURES))
SLOT_KINDS = ["EXPLORE"] * 6 + ["VERIFY"] * 2 + ["RETURN_HOME", "RETURN_TO_LINK", "HOLD"]
K_SLOTS = len(SLOT_KINDS)


class Cand(object):
    """One candidate action. goal is (x, y, yaw) or None (HOLD in place)."""

    def __init__(self, kind, goal=None, node=-1):
        self.search = False
        self.visited = False
        self.kind = kind
        self.goal = goal
        self.node = node
        self.d_r = INF
        self.d_h = INF
        self.d_l = INF
        self.eu = 0.0
        self.size_m = 0.0
        self.unknown_frac = 0.0
        self.openness = 0.0
        self.utility = 0.0
        self.feasible = True
        self.feas_route = True
        self.why_not = ""
        self.match = False
        self.is_prev = False
        self.sighting = None
        self.hits = 0
        self.rank = None
        self.bat_after = 0.0
        self.time_margin = 1.0
        self.dist_margin = 1.0
        self.feat = None
        self.allowed = False       # may the learned policy pick it (set by the arbiter)
        self.frontier = None


class Ctx(object):
    pass


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _sector(theta):
    d = math.degrees(theta)
    if -30.0 <= d <= 30.0:
        return "STRAIGHT"
    if 30.0 < d < 150.0:
        return "LEFT"
    if -150.0 < d < -30.0:
        return "RIGHT"
    return "BACK"


def parse_persons(obs):
    out = []
    for p in obs.get("persons") or ():
        try:
            out.append({"x": float(p["x"]), "y": float(p["y"]),
                        "confirmed": bool(p.get("confirmed", False)),
                        "reported": bool(p.get("reported", False)),
                        "hits": int(p.get("hits", 1) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _los(view, ax, ay, bx, by):
    """Straight line a->b crosses no occupied coarse cell."""
    L = math.hypot(bx - ax, by - ay)
    n = max(2, int(L / (0.5 * view.cell)) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = ax + (bx - ax) * t
    ys = ay + (by - ay) * t
    ii = np.floor((ys - view.y0) / view.cell).astype(np.int64)
    jj = np.floor((xs - view.x0) / view.cell).astype(np.int64)
    ok = (ii >= 0) & (ii < view.shape[0]) & (jj >= 0) & (jj < view.shape[1])
    if not ok.all():
        return False
    return not view.occ[ii, jj].any()


def standoff(ctx, px, py, rmin=0.8, rmax=2.5):
    """Reachable cell near a sighting with line of sight to it: closest to the robot, preferring
    ~verify_standoff_m away. Returns (i, j) or None."""
    view = ctx.view
    cell = view.cell
    ci, cj = view.cell_of(px, py)
    R = int(math.ceil(rmax / cell))
    H, W = view.shape
    a0, a1 = max(0, ci - R), min(H, ci + R + 1)
    b0, b1 = max(0, cj - R), min(W, cj + R + 1)
    if a0 >= a1 or b0 >= b1:
        return None
    win = ctx.reach[a0:a1, b0:b1]
    wi, wj = np.nonzero(np.isfinite(win))
    if wi.size == 0:
        return None
    wi = wi + a0
    wj = wj + b0
    gx, gy = view.xy_of(wi, wj)
    dd = np.hypot(gx - px, gy - py)
    m = (dd >= rmin) & (dd <= rmax)
    if not m.any():
        m = dd <= rmax
        if not m.any():
            return None
    wi, wj, gx, gy, dd = wi[m], wj[m], gx[m], gy[m], dd[m]
    cost = ctx.reach[wi, wj] * cell + 2.0 * np.abs(dd - ctx.P["verify_standoff_m"])
    order = np.argsort(cost)[:40]
    for k in order:
        if _los(view, float(gx[k]), float(gy[k]), px, py):
            return (int(wi[k]), int(wj[k]))
    return None


def arrival_yaw(ctx, node, default):
    """Heading on arrival at `node` along the geodesic from the robot (last ~1 m of the path)."""
    path = descend(ctx.graph, ctx.D[0], node, max_steps=8)
    if len(path) < 2:
        return default
    a, b = path[-1], path[0]
    x0, y0 = ctx.view.xy_of(ctx.graph.rr[a], ctx.graph.cc[a])
    x1, y1 = ctx.view.xy_of(ctx.graph.rr[b], ctx.graph.cc[b])
    if abs(x1 - x0) + abs(y1 - y0) < 1e-6:
        return default
    return math.atan2(float(y1 - y0), float(x1 - x0))


def build_context(obs, M, P, S):
    """obs + mission + params + brain memory -> Ctx with candidates (features filled later)."""
    c = Ctx()
    c.t0 = clock()
    c.M, c.P, c.S = M, P, S
    pose = obs["pose"]
    c.rx, c.ry = float(pose[0]), float(pose[1])
    c.ryaw = float(pose[2]) if len(pose) > 2 and pose[2] is not None else 0.0
    home = obs.get("home") or (c.rx, c.ry)
    c.hx, c.hy = float(home[0]), float(home[1])
    c.battery = fnum(obs.get("battery_pct"), 100.0)
    c.link_ok = bool(obs.get("link_ok", True))
    c.since = fnum(obs.get("since_contact_s"), 0.0)
    c.elapsed = fnum(obs.get("elapsed_s"), 0.0)
    c.odo = fnum(obs.get("odo_m"), 0.0)
    c.persons = parse_persons(obs)
    llp = obs.get("last_link_pose")
    c.llp = (float(llp[0]), float(llp[1])) if llp is not None else None
    c.trail = [(float(p[0]), float(p[1])) for p in (obs.get("trail") or ())]
    c.blocked_in = list(obs.get("blocked") or ())

    include = [(c.rx, c.ry), (c.hx, c.hy)] + ([c.llp] if c.llp else []) + \
        [(p["x"], p["y"]) for p in c.persons]
    view = MapView(obs["grid"], obs["res"], obs.get("origin", (0.0, 0.0)), P["cell_m"],
                   P["inflate_m"], P["occ_thresh"], P["free_thresh"], c.blocked_in, include)
    c.view = view
    cell = view.cell
    t_map = clock()

    # --- anchor cells ---------------------------------------------------------------------
    view.open_escape(c.rx, c.ry, P["escape_m"])
    rc = view.snap(c.rx, c.ry, P["robot_snap_m"], view.passable)
    if rc is None:
        i, j = view.cell_of(c.rx, c.ry)
        rc = (i, j) if view.force_passable(i, j) else None
    hc = view.snap(c.hx, c.hy, 1.5, view.passable)
    if hc is None:
        i, j = view.cell_of(c.hx, c.hy)
        if view.inside(i, j) and not view.occ[i, j] and not view.blocked[i, j] and \
                (not view.unknown[i, j] or math.hypot(c.hx - c.rx, c.hy - c.ry) < 1.0):
            hc = (i, j) if view.force_passable(i, j) else None
    lc = None
    need_link = c.llp is not None and not c.link_ok
    if need_link:
        lc = view.snap(c.llp[0], c.llp[1], 1.5, view.passable)
    c.robot_cell, c.home_cell, c.link_cell = rc, hc, lc

    graph = Graph(view.passable)
    c.graph = graph
    rn, hn, ln = graph.node(rc), graph.node(hc), graph.node(lc)
    c.rn, c.hn, c.ln = rn, hn, ln
    c.D = distances(graph, [rn, hn, ln])
    t_geo = clock()
    c.reach = graph.image(c.D[0])          # in cells
    c.d_home_now = float(c.D[0][hn]) * cell if hn >= 0 and rn >= 0 else INF
    c.home_reachable = bool(np.isfinite(c.d_home_now))
    c.d_link_now = float(c.D[0][ln]) * cell if ln >= 0 and rn >= 0 else INF
    if ln >= 0 and np.isfinite(c.D[1][ln]):
        c.d_home_link = float(c.D[1][ln]) * cell
    elif c.llp is not None:
        c.d_home_link = math.hypot(c.llp[0] - c.hx, c.llp[1] - c.hy)
    else:
        c.d_home_link = 0.0
    c.at_home = (c.home_reachable and c.d_home_now <= P["home_radius_m"]) or \
        math.hypot(c.rx - c.hx, c.ry - c.hy) <= P["home_radius_m"]

    # --- energy / time / distance model ---------------------------------------------------
    c.speed = P["slow_speed_mps"] if M["speed"] == "SLOW" else P["speed_mps"]
    rate = P["drain_pct_per_m"] + P["drain_pct_per_s"] / c.speed
    b0, o0 = S.get("bat0"), S.get("odo0")
    c.rate_model = rate
    if b0 is not None and c.odo - o0 >= P["online_rate_min_m"]:
        obs_rate = (b0 - c.battery) / max(c.odo - o0, 1e-6)
        rate = max(rate, obs_rate)
    c.rate = rate
    c.pf = P["path_factor"]
    c.floor = max(P["battery_floor_pct"], M["min_battery_pct"] or 0.0)
    c.la_m = c.speed * P["redecide_s"]
    c.reserve_now = c.reserve(c.d_home_now) if c.home_reachable else INF
    c.limit_s = M["max_duration_min"] * 60.0 if M["max_duration_min"] else None
    c.limit_m = M["max_distance_m"] if M["max_distance_m"] else None
    c.start_yaw = S.get("start_yaw", c.ryaw)
    c.branch_pending = M["first_branch"] != "ANY" and not S.get("branch_committed", False)

    # --- frontiers ------------------------------------------------------------------------
    frs, fmask = find_frontiers(view, c.reach, (c.rx, c.ry), P["goal_snap_m"],
                                P["min_frontier_cells"], P["frontier_split_m"])
    c.frontier_mask = fmask
    c.n_frontiers_all = len(frs)
    c.frontier_len = float(sum(f.size_m for f in frs))
    t_fr = clock()

    ii_unk = view.integral("unknown")
    ii_pas = view.integral("passable")
    r_unk = int(round(3.0 / cell))
    r_open = int(round(1.7 / cell))

    # frontier goals the robot already stood on without resolving them (sensor shadow, a gap
    # too narrow to see through) are not offered again: otherwise the loop can pick them forever
    vg = S.setdefault("visited_goals", [])
    pg = S.get("prev_goal")
    if S.get("prev_kind") == "EXPLORE" and pg is not None and math.hypot(c.rx - pg[0], c.ry - pg[1]) <= 0.6:
        vg.append((float(pg[0]), float(pg[1])))
        del vg[:-300]
    vga = np.asarray(vg, dtype=np.float64).reshape(-1, 2)
    cands = []
    for f in frs:
        n = graph.node(f.goal_cell)
        cd = Cand("EXPLORE", (f.gx, f.gy, f.gyaw), n)
        cd.frontier = f
        cd.size_m = f.size_m
        cd.visited = bool(vga.shape[0]) and bool(np.any(np.hypot(vga[:, 0] - f.gx, vga[:, 1] - f.gy) <= 1.0))
        cands.append(cd)
    c.frontiers = cands

    # --- camera coverage and search goals ---------------------------------------------------
    tm = S.setdefault("trail_mem", {})
    for (x, y) in c.trail + [(c.rx, c.ry)]:
        tm[(int(round(x * 2.0)), int(round(y * 2.0)))] = (x, y)      # one point per 0.5 m
    c.covered = np.zeros(view.shape, dtype=bool)
    if tm:
        pts = np.array(list(tm.values()), dtype=np.float64)
        ti = np.floor((pts[:, 1] - view.y0) / cell).astype(np.int64)
        tj = np.floor((pts[:, 0] - view.x0) / cell).astype(np.int64)
        ok = (ti >= 0) & (ti < view.shape[0]) & (tj >= 0) & (tj < view.shape[1])
        seeds = graph.idx[ti[ok], tj[ok]]
        seeds = seeds[seeds < graph.N]
        if seeds.size:
            dc = distances(graph, [seeds], max_dist=P["cam_cover_m"] / cell)[0]
            fin = np.isfinite(dc)
            c.covered[graph.rr[fin], graph.cc[fin]] = True
    reach_fin = np.isfinite(c.reach)
    c.cam_covered = float(np.count_nonzero(c.covered & reach_fin)) / max(1.0, float(np.count_nonzero(reach_fin)))
    c.searches = []
    if M["task"] == "SEARCH_SURVIVORS":
        for sg in search_goals(view, c.reach, c.covered, fmask, (c.rx, c.ry), P["search_min_area_m2"]):
            n = graph.node(sg.goal_cell)
            cd = Cand("EXPLORE", (sg.gx, sg.gy, sg.gyaw), n)
            cd.search = True
            cd.size_m = sg.area_m2
            cd.visited = bool(vga.shape[0]) and bool(np.any(np.hypot(vga[:, 0] - sg.gx, vga[:, 1] - sg.gy) <= 1.0))
            c.searches.append(cd)
    c.explore = c.frontiers + c.searches
    t_cov = clock()

    # --- persons ---------------------------------------------------------------------------
    c.confirmed = [p for p in c.persons if p["confirmed"]]
    c.unreported = [p for p in c.confirmed if not p["reported"]]
    c.open_sightings = []
    for p in c.persons:
        if p["confirmed"]:
            continue
        rec = _sighting_record(S, p)
        if rec["given_up"]:
            continue
        c.open_sightings.append((p, rec))
    c.open_sightings.sort(key=lambda pr: math.hypot(pr[0]["x"] - c.rx, pr[0]["y"] - c.ry))
    c.verify = []
    for p, rec in c.open_sightings[:P["k_verify"]]:
        sc = standoff(c, p["x"], p["y"])
        if sc is None:
            continue
        gx, gy = view.xy_of(sc[0], sc[1])
        cd = Cand("VERIFY", (float(gx), float(gy), math.atan2(p["y"] - gy, p["x"] - gx)), graph.node(sc))
        cd.sighting = (p, rec)
        cd.hits = p["hits"]
        c.verify.append(cd)

    # --- fixed candidates -----------------------------------------------------------------
    c.home_cand = None
    if hn >= 0 and rn >= 0 and np.isfinite(c.D[0][hn]):
        hi, hj = view.cell_of(c.hx, c.hy)
        if view.inside(hi, hj) and view.passable[hi, hj]:
            hx, hy = c.hx, c.hy
        else:
            hx, hy = view.xy_of(hc[0], hc[1])
        c.home_cand = Cand("RETURN_HOME", (float(hx), float(hy), 0.0), hn)
    c.link_cand = None
    if need_link and ln >= 0 and rn >= 0 and np.isfinite(c.D[0][ln]):
        lx, ly = view.xy_of(lc[0], lc[1])
        c.link_cand = Cand("RETURN_TO_LINK", (float(lx), float(ly), 0.0), ln)
    c.hold_cand = Cand("HOLD", None, rn)

    allc = c.explore + c.verify + [x for x in (c.home_cand, c.link_cand) if x is not None] + [c.hold_cand]
    for cd in allc:
        _fill_geometry(c, cd, ii_unk, ii_pas, r_unk, r_open)
    for cd in c.verify + [x for x in (c.home_cand, c.link_cand) if x is not None]:
        if cd.goal is not None and cd.node >= 0:
            yaw = arrival_yaw(c, cd.node, cd.goal[2])
            if cd.kind != "VERIFY":
                cd.goal = (cd.goal[0], cd.goal[1], yaw)
    _feasibility(c)
    _utilities(c)
    c.t_map_ms = (t_map - c.t0) * 1e3
    c.t_geo_ms = (t_geo - t_map) * 1e3
    c.t_fr_ms = (t_fr - t_geo) * 1e3
    c.t_cov_ms = (t_cov - t_fr) * 1e3
    return c


def _sighting_record(S, p):
    """The brain's memory about a sighting, matched by position (the executive re-sends the
    whole list every time and may move a sighting a little as hits accumulate)."""
    recs = S.setdefault("sightings", [])
    best, bd = None, 1.5
    for r in recs:
        d = math.hypot(r["x"] - p["x"], r["y"] - p["y"])
        if d < bd:
            best, bd = r, d
    if best is None:
        best = {"x": p["x"], "y": p["y"], "attempts": 0, "given_up": False}
        recs.append(best)
    else:
        best["x"], best["y"] = p["x"], p["y"]
    return best


def _reserve(self, d_home):
    return (d_home * self.pf * self.rate) * (1.0 + self.P["reserve_frac"]) + self.P["reserve_pct"]


def _energy(self, d):
    return d * self.pf * self.rate


Ctx.reserve = _reserve
Ctx.energy = _energy


def _fill_geometry(c, cd, ii_unk, ii_pas, r_unk, r_open):
    cell = c.view.cell
    if cd.kind == "HOLD" or cd.goal is None:
        cd.d_r = 0.0
        cd.d_h = c.d_home_now
        cd.d_l = c.d_link_now
        gx, gy = c.rx, c.ry
        cellij = c.robot_cell
    else:
        n = cd.node
        cd.d_r = float(c.D[0][n]) * cell if n >= 0 else INF
        cd.d_h = float(c.D[1][n]) * cell if n >= 0 else INF
        cd.d_l = float(c.D[2][n]) * cell if n >= 0 else INF
        gx, gy = cd.goal[0], cd.goal[1]
        cellij = c.view.cell_of(gx, gy)
    cd.eu = math.hypot(gx - c.rx, gy - c.ry)
    if cellij is not None and c.view.inside(cellij[0], cellij[1]):
        u, a = box_count(ii_unk, cellij[0], cellij[1], r_unk)
        cd.unknown_frac = float(u) / max(float(a), 1.0)
        o, a = box_count(ii_pas, cellij[0], cellij[1], r_open)
        cd.openness = float(o) / max(float(a), 1.0)
    th = math.atan2(gy - c.ry, gx - c.rx)
    cd.bear = wrap(th - c.ryaw) if cd.eu > 0.05 else 0.0
    cd.bear0 = wrap(th - c.start_yaw) if cd.eu > 0.05 else 0.0
    cd.sector = _sector(cd.bear0) if cd.eu > 0.05 else "HERE"
    pg = c.S.get("prev_goal")
    cd.is_prev = pg is not None and cd.goal is not None and \
        math.hypot(cd.goal[0] - pg[0], cd.goal[1] - pg[1]) <= 1.5


def _feasibility(c):
    """Battery, time, distance and link-silence feasibility of every candidate with a goal."""
    M, P = c.M, c.P
    ret = M["return_when_done"]
    tm = P["time_margin_frac"]
    dwell_e = P["drain_pct_per_s"] * P["dwell_s"]
    for cd in c.explore + c.verify + [x for x in (c.home_cand, c.link_cand) if x is not None]:
        why = []
        if not np.isfinite(cd.d_r):
            why.append("unreachable")
        if getattr(cd, "visited", False):
            why.append("visited")
        d_back = cd.d_h
        if not np.isfinite(d_back):
            # no known route home from there: optimistic straight-line x tortuosity estimate
            d_back = math.hypot(cd.goal[0] - c.hx, cd.goal[1] - c.hy) * P["route_tortuosity"]
        extra = dwell_e if cd.kind in ("EXPLORE", "VERIFY") else 0.0
        if cd.kind == "RETURN_HOME":
            cd.bat_after = c.battery - c.energy(cd.d_r) - c.floor
        else:
            cd.bat_after = c.battery - c.energy(cd.d_r) - extra - c.reserve(d_back) - c.floor
        if cd.kind in ("EXPLORE", "VERIFY") and cd.bat_after < 0:
            why.append("battery")
        if c.limit_s is not None:
            need = c.elapsed + cd.d_r * c.pf / c.speed + (P["dwell_s"] if cd.kind != "RETURN_HOME" else 0.0)
            if ret and cd.kind != "RETURN_HOME":
                need += d_back * (1.0 + tm) * c.pf / c.speed
            cd.time_margin = (c.limit_s - need) / c.limit_s
            if cd.kind in ("EXPLORE", "VERIFY") and cd.time_margin < 0:
                why.append("time")
        if c.limit_m is not None:
            need = c.odo + cd.d_r * c.pf
            if ret and cd.kind != "RETURN_HOME":
                need += d_back * (1.0 + tm) * c.pf
            cd.dist_margin = (c.limit_m - need) / c.limit_m
            if cd.kind in ("EXPLORE", "VERIFY") and cd.dist_margin < 0:
                why.append("distance")
        if cd.kind == "EXPLORE" and not c.link_ok and M["on_link_loss"] == "CONTINUE_THEN_RETURN" \
                and np.isfinite(cd.d_l):
            if c.since + (cd.d_r + cd.d_l) * c.pf / c.speed > P["link_grace_s"]:
                why.append("link silence")
        cd.feasible = not why
        cd.why_not = ",".join(why)
        # when the way home is gone, only reachability and the battery may stop the search for
        # another one: the mission's time/distance budget is about exploring, not getting home
        cd.feas_route = [w for w in why if w not in ("time", "distance", "link silence")] == []


def _utilities(c):
    """Rules frontier utility: gain / cost, with the first-branch and persistence biases."""
    P = c.P
    for cd in c.explore:
        if not np.isfinite(cd.d_r):
            cd.utility = 0.0
            continue
        if getattr(cd, "search", False):
            gain = P["search_gain"] * min(cd.size_m, 30.0) * (1.3 if cd.openness > 0.5 else 1.0)
        else:
            gain = min(cd.size_m, 4.0) + 2.0 * cd.unknown_frac
        u = gain / (cd.d_r + 2.0)
        cd.match = c.branch_pending and cd.sector == c.M["first_branch"]
        if cd.match:
            u *= P["first_branch_gain"]
        if cd.is_prev:
            u *= P["persist_gain"]
        cd.utility = u
    order = sorted([cd for cd in c.explore if cd.feasible], key=lambda x: -x.utility)
    for r, cd in enumerate(order):
        cd.rank = r


def candidate_slots(c):
    """Fixed slot layout for the learned policy: 6 frontiers, 2 verify, home, link, hold."""
    fr = sorted([cd for cd in c.explore if cd.feasible], key=lambda x: -x.utility)[:6]
    slots = [None] * K_SLOTS
    for k, cd in enumerate(fr):
        slots[k] = cd
    for k, cd in enumerate(c.verify[:2]):
        slots[6 + k] = cd
    slots[8] = c.home_cand
    slots[9] = c.link_cand
    slots[10] = c.hold_cand
    return slots


def cand_features(c, cd, rules_pick=False):
    f = np.zeros(NC, dtype=np.float32)
    k = KINDS.index(cd.kind)
    f[k] = 1.0
    fin = lambda v, cap: cap if not np.isfinite(v) else v
    f[CI["d_robot"]] = _clip(fin(cd.d_r, 150.0) / 50.0, 0.0, 3.0)
    f[CI["d_home"]] = _clip(fin(cd.d_h, 150.0) / 50.0, 0.0, 3.0)
    f[CI["d_euclid"]] = _clip(cd.eu / 50.0, 0.0, 3.0)
    f[CI["detour"]] = _clip(fin(cd.d_r, 150.0) / max(cd.eu, 1.0) / 3.0, 0.0, 1.0)
    f[CI["sin_bear"]] = math.sin(cd.bear)
    f[CI["cos_bear"]] = math.cos(cd.bear)
    f[CI["sin_bear0"]] = math.sin(cd.bear0)
    f[CI["cos_bear0"]] = math.cos(cd.bear0)
    f[CI["branch_match"]] = 1.0 if (c.branch_pending and cd.sector == c.M["first_branch"]) else 0.0
    if getattr(cd, "search", False):
        f[CI["is_search"]] = 1.0
        f[CI["size"]] = _clip(cd.size_m / 10.0, 0.0, 2.0)
    else:
        f[CI["size"]] = _clip(cd.size_m / 5.0, 0.0, 2.0)
    f[CI["unknown_frac"]] = cd.unknown_frac
    f[CI["openness"]] = cd.openness
    f[CI["bat_margin"]] = _clip(cd.bat_after / 100.0, -1.0, 1.0) if cd.kind != "HOLD" else \
        _clip((c.battery - c.floor - (c.reserve_now if np.isfinite(c.reserve_now) else 100.0)) / 100.0, -1.0, 1.0)
    f[CI["time_margin"]] = _clip(cd.time_margin, -1.0, 1.0)
    f[CI["dist_margin"]] = _clip(cd.dist_margin, -1.0, 1.0)
    f[CI["utility"]] = math.log1p(10.0 * cd.utility) / 3.0 if cd.kind == "EXPLORE" else 0.0
    f[CI["rules_pick"]] = 1.0 if rules_pick else 0.0
    f[CI["is_prev_goal"]] = 1.0 if cd.is_prev else 0.0
    dh_now = c.d_home_now if np.isfinite(c.d_home_now) else 150.0
    f[CI["d_home_delta"]] = _clip((fin(cd.d_h, 150.0) - dh_now) / 50.0, -2.0, 2.0)
    if c.llp is not None and not c.link_ok:
        f[CI["beyond_link"]] = _clip((fin(cd.d_h, 150.0) - c.d_home_link) / 50.0, -2.0, 2.0)
    gx, gy = (cd.goal[0], cd.goal[1]) if cd.goal is not None else (c.rx, c.ry)
    if c.open_sightings:
        d = min(math.hypot(p["x"] - gx, p["y"] - gy) for p, _ in c.open_sightings)
        f[CI["person_near"]] = math.exp(-d / 3.0)
    if c.trail:
        tr = np.asarray(c.trail, dtype=np.float64)
        f[CI["visited"]] = float(np.mean(np.hypot(tr[:, 0] - gx, tr[:, 1] - gy) <= 2.0))
    if c.view.blocked_list:
        d = min(max(0.0, math.hypot(bx - gx, by - gy) - br) for bx, by, br in c.view.blocked_list)
        f[CI["blocked_near"]] = math.exp(-d / 3.0)
    f[CI["rank"]] = (cd.rank / 6.0) if (cd.kind == "EXPLORE" and cd.rank is not None) else 1.0
    f[CI["hits"]] = _clip(cd.hits / 5.0, 0.0, 1.0)
    return f


def global_features(c):
    g = np.zeros(NG, dtype=np.float32)
    M = c.M
    g[GI["battery"]] = c.battery / 100.0
    rn = c.reserve_now if np.isfinite(c.reserve_now) else 100.0
    g[GI["reserve_margin"]] = _clip((c.battery - c.floor - rn) / 100.0, -1.0, 1.0)
    g[GI["link_ok"]] = 1.0 if c.link_ok else 0.0
    g[GI["since_contact"]] = _clip(c.since / 600.0, 0.0, 1.0)
    g[GI["elapsed"]] = _clip(c.elapsed / 3600.0, 0.0, 1.5)
    if c.limit_s:
        g[GI["time_frac"]] = _clip(c.elapsed / c.limit_s, 0.0, 1.5)
        g[GI["has_time_limit"]] = 1.0
    g[GI["odo"]] = _clip(c.odo / 1000.0, 0.0, 1.5)
    if c.limit_m:
        g[GI["dist_frac"]] = _clip(c.odo / c.limit_m, 0.0, 1.5)
        g[GI["has_dist_limit"]] = 1.0
    g[GI["d_home_now"]] = _clip(c.d_home_now / 100.0, 0.0, 2.0) if c.home_reachable else 2.0
    g[GI["home_reachable"]] = 1.0 if c.home_reachable else 0.0
    g[GI["found"]] = _clip(len(c.confirmed) / 3.0, 0.0, 1.0)
    g[GI["unreported"]] = _clip(len(c.unreported) / 3.0, 0.0, 1.0)
    g[GI["unconfirmed"]] = _clip(len(c.open_sightings) / 3.0, 0.0, 1.0)
    cell2 = c.view.cell ** 2
    g[GI["known_free"]] = _clip(float(np.count_nonzero(c.view.free)) * cell2 / 2000.0, 0.0, 1.5)
    g[GI["cam_covered"]] = c.cam_covered
    g[GI["n_search"]] = _clip(len(c.searches) / 10.0, 0.0, 1.5)
    g[GI["n_frontiers"]] = _clip(c.n_frontiers_all / 10.0, 0.0, 1.5)
    g[GI["frontier_len"]] = _clip(c.frontier_len / 20.0, 0.0, 1.5)
    g[GI["n_blocked"]] = _clip(len(c.view.blocked_list) / 3.0, 0.0, 1.0)
    g[GI["link_known"]] = 1.0 if c.llp is not None else 0.0
    if c.llp is not None and not c.link_ok and np.isfinite(c.d_link_now):
        g[GI["d_link"]] = _clip(c.d_link_now / 50.0, 0.0, 2.0)
    g[GI["branch_pending"]] = 1.0 if c.branch_pending else 0.0
    g[GI["min_battery"]] = (M["min_battery_pct"] or 0.0) / 100.0
    g[GI["speed_slow"]] = 1.0 if M["speed"] == "SLOW" else 0.0
    g[GI["return_when_done"]] = 1.0 if M["return_when_done"] else 0.0
    g[GI["report_person"]] = 1.0 if M["report_person"] else 0.0
    g[GI["approach_person"]] = 1.0 if M["approach_person"] else 0.0
    if M["task"] in TASK_ORDER:
        g[GI["task_" + M["task"].lower()]] = 1.0
    g[GI["opf_" + M["on_person_found"].lower()]] = 1.0
    g[GI["oll_" + M["on_link_loss"].lower()]] = 1.0
    g[GI["fb_" + M["first_branch"].lower()]] = 1.0
    return g


def feature_doc():
    lines = ["# DHWANI brain features", "",
             "Generated from `dhwani_brain/features.py` (`python -m dhwani_brain.features`).",
             "Every value is computed on the rover from the `obs` dict alone (plus the brain's own",
             "memory of earlier obs dicts: start heading, first-branch commitment, verify attempts,",
             "previous goal, battery and odometry at the first call).", "",
             "Distances are geodesic (octile wavefront on the 0.2 m planning grid, obstacles and",
             "blocked regions inflated by 0.15 m) unless marked straight-line. Energy uses the",
             "brain's model: pct per metre = drain_pct_per_m + drain_pct_per_s / speed, times",
             "path_factor, raised to the measured rate once 20 m of odometry exist.", "",
             "## Per-candidate vector (%d values)" % NC, "",
             "Slots: 6 frontier goals (top by rules utility, feasible only), 2 verify standoff",
             "points, RETURN_HOME, RETURN_TO_LINK, HOLD. Missing slots are masked.", "",
             "| # | name | meaning and normalisation |", "|---|---|---|"]
    for i, (n, d) in enumerate(CAND_FEATURES):
        lines.append("| %d | `%s` | %s |" % (i, n, d))
    lines += ["", "## Global vector (%d values)" % NG, "",
              "| # | name | meaning and normalisation |", "|---|---|---|"]
    for i, (n, d) in enumerate(GLOBAL_FEATURES):
        lines.append("| %d | `%s` | %s |" % (i, n, d))
    lines += ["", "The learned network additionally standardises each input with the mean and",
              "standard deviation stored in its .npz (computed on training data only)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(feature_doc(), end="")
