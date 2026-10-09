# -*- coding: utf-8 -*-
"""Deterministic rules policy: the mission semantics, in a fixed order.

    1. safety      route home exists, battery reserve, time and distance limits
    2. orders      RETURN_HOME / HALT tasks, and returns already latched by an earlier rule
    3. link loss   on_link_loss
    4. persons     report, return, stay, verify sightings
    5. explore     best feasible frontier by utility, with the first_branch bias
    6. done        return home or stop when nothing feasible is left

Every decision carries a tier. 'safety' and 'mission' decisions are final. Only 'discretionary'
ones (which frontier, whether to dwell, when exploring is no longer worth it) may be replaced by
the learned policy, and only by another candidate that already passed the same feasibility
checks. That is what "safety overrides always come from rules" means in code.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

SAFETY, MISSION, DISCRETIONARY = "safety", "mission", "discretionary"


class Decision(object):
    def __init__(self, kind, cand=None, goal=None, sensor="MOVE", reason="", tier=MISSION, extra=None):
        self.kind = kind
        self.cand = cand
        self.goal = goal if goal is not None else (cand.goal if cand is not None else None)
        self.sensor = sensor
        self.reason = reason
        self.tier = tier
        self.extra = extra or {}
        self.source = "rules"
        self.confidence = 1.0


def _xy(p):
    return "(%.1f, %.1f)" % (p[0], p[1])


def _hold(reason, tier, goal=None, cand=None):
    return Decision("HOLD", cand, goal, "MOVE", reason, tier)


def _home(c, reason, tier):
    if c.home_cand is None:
        return None
    return Decision("RETURN_HOME", c.home_cand, None, "MOVE", reason, tier)


def _return_or_done(c, why, tier):
    if c.at_home:
        return Decision("DONE", None, None, "MOVE", why + "; already home", tier)
    d = _home(c, why + "; returning home (%.0f m)" % c.d_home_now, tier)
    return d if d is not None else route_blocked(c, why)


def _latch(c, why):
    c.S["latched"] = why
    return _return_or_done(c, why, MISSION)


def _to_link(c, why, tier=MISSION):
    """Go back to where the radio last worked; if we are there and it still does not, go home."""
    P = c.P
    if c.link_cand is not None and c.d_link_now > P["link_radius_m"]:
        return Decision("RETURN_TO_LINK", c.link_cand, None, "MOVE",
                        why + "; going back to the last radio contact point %s (%.0f m)"
                        % (_xy(c.llp), c.d_link_now), tier)
    if c.link_cand is not None:
        why += "; no link even at the last contact point"
    else:
        why += "; last contact point unknown or unreachable"
    return _return_or_done(c, why, tier)


def safe_spot(c, radius_m=6.0):
    """Most open reachable cell within radius_m that is not near a blocked region."""
    view = c.view
    ii = view.integral("passable")
    near = c.reach * view.cell <= radius_m
    near &= ~view.near_blocked
    wi, wj = np.nonzero(near)
    if wi.size == 0:
        return None
    r = max(1, int(round(1.0 / view.cell)))
    h, w = view.shape
    a0 = np.clip(wi - r, 0, h)
    a1 = np.clip(wi + r + 1, 0, h)
    b0 = np.clip(wj - r, 0, w)
    b1 = np.clip(wj + r + 1, 0, w)
    open_ = ii[a1, b1] - ii[a0, b1] - ii[a1, b0] + ii[a0, b0]
    score = open_.astype(np.float64) - 0.3 * c.reach[wi, wj]
    k = int(np.argmax(score))
    x, y = view.xy_of(wi[k], wj[k])
    return (float(x), float(y), c.ryaw), float(c.reach[wi[k], wj[k]] * view.cell)


def route_blocked(c, why="no known route home"):
    """Home is unreachable on the known map (rockfall, failed plans). Look for another way only
    within the reserve; else get to radio contact; else hold somewhere safe and keep the data."""
    P = c.P
    nb = len(c.view.blocked_list)
    why = why + (" (%d blocked region%s)" % (nb, "" if nb == 1 else "s") if nb else "")
    opts = [f for f in c.frontiers if f.feas_route and np.isfinite(f.d_r)]
    if opts:
        tort = P["route_tortuosity"]
        best = min(opts, key=lambda f: f.d_r + tort * math.hypot(f.goal[0] - c.hx, f.goal[1] - c.hy))
        est = best.d_r + tort * math.hypot(best.goal[0] - c.hx, best.goal[1] - c.hy)
        return Decision("EXPLORE", best, None, "MOVE",
                        "%s; exploring for another way home via the frontier at %s "
                        "(route estimate %.0f m, fits the battery reserve)" % (why, _xy(best.goal), est),
                        SAFETY)
    if not c.link_ok and c.link_cand is not None and c.d_link_now > P["link_radius_m"]:
        return Decision("RETURN_TO_LINK", c.link_cand, None, "MOVE",
                        "%s and no frontier fits the reserve; going to the last radio contact point "
                        "%s to report and wait" % (why, _xy(c.llp)), SAFETY)
    spot = safe_spot(c)
    if spot is not None and spot[1] > 0.6:
        return _hold("%s and no way to find one within the battery reserve; moving %.1f m to an "
                     "open spot away from the blockage to hold and keep the data" % (why, spot[1]),
                     SAFETY, goal=spot[0])
    return _hold("%s and no way to find one within the battery reserve; holding here and "
                 "keeping the data" % why, SAFETY)


def decide(c):
    M, P, S = c.M, c.P, c.S
    task = M["task"]

    # ---- 0. preconditions ----------------------------------------------------------------
    if c.rn < 0:
        return _hold("robot pose is not on known ground of the map; holding", SAFETY)
    if task == "UNSUPPORTED" or not M["understood"]:
        return _hold("instruction not understood; holding for a clearer order", MISSION)
    if task == "HALT":
        return _hold("HALT ordered", MISSION)

    # ---- 1. safety: route home, battery reserve, limits --------------------------------------
    if not c.home_reachable and not c.at_home:
        return route_blocked(c)
    need = c.reserve(c.d_home_now + c.la_m) + c.energy(c.la_m) + c.floor
    if c.battery <= need:
        return _return_or_done(c, "battery %.0f%% is at the return threshold %.0f%% (energy home over "
                               "%.0f m plus margin, floor %.0f%%)" % (c.battery, need, c.d_home_now, c.floor),
                               SAFETY)
    tm = P["time_margin_frac"]
    if c.limit_s is not None:
        if M["return_when_done"]:
            t_home = (c.d_home_now + c.la_m) * c.pf / c.speed * (1.0 + tm)
            if c.elapsed + P["redecide_s"] + t_home >= c.limit_s:
                return _return_or_done(c, "time limit: %.0f of %.0f min used and the way home takes about "
                                       "%.0f min" % (c.elapsed / 60.0, c.limit_s / 60.0, t_home / 60.0), SAFETY)
        elif c.elapsed >= c.limit_s:
            return Decision("DONE", None, None, "MOVE", "time limit of %.0f min reached" % (c.limit_s / 60.0),
                            SAFETY)
    if c.limit_m is not None:
        if M["return_when_done"]:
            d_need = c.odo + c.la_m + (c.d_home_now + c.la_m) * c.pf * (1.0 + tm)
            if d_need >= c.limit_m:
                return _return_or_done(c, "distance limit: %.0f of %.0f m driven and home is %.0f m away"
                                       % (c.odo, c.limit_m, c.d_home_now), SAFETY)
        elif c.odo >= c.limit_m:
            return Decision("DONE", None, None, "MOVE", "distance limit of %.0f m reached" % c.limit_m, SAFETY)

    # ---- 2. orders and latched returns -------------------------------------------------------
    if task == "RETURN_HOME":
        return _return_or_done(c, "RETURN_HOME ordered", MISSION)
    if S.get("latched"):
        return _return_or_done(c, S["latched"], MISSION)

    # ---- 3. link-loss policy -----------------------------------------------------------------
    if not c.link_ok:
        oll = M["on_link_loss"]
        if oll == "RETURN_NOW":
            return _latch(c, "link lost %.0f s ago and the mission says return now" % c.since)
        if oll == "HOLD":
            return _hold("link lost %.0f s ago; the mission says hold here until it comes back" % c.since,
                         MISSION)
        if oll == "CONTINUE_THEN_RETURN" and c.since >= P["link_grace_s"]:
            return _to_link(c, "no contact for %.0f s (limit %.0f s)" % (c.since, P["link_grace_s"]))

    # ---- 4. person policy --------------------------------------------------------------------
    opf = M["on_person_found"]
    if M["report_person"] and c.unreported and not c.link_ok:
        p = min(c.unreported, key=lambda q: math.hypot(q["x"] - c.rx, q["y"] - c.ry))
        if opf == "REPORT_AND_RETURN":
            return _latch(c, "person found at %s and the mission says report and return; the report "
                             "goes out as soon as the link is back on the way" % _xy((p["x"], p["y"])))
        return _to_link(c, "person found at %s is not reported yet (no link)" % _xy((p["x"], p["y"])))
    if opf == "REPORT_AND_RETURN" and c.confirmed:
        return _latch(c, "person found and reported; the mission says return")
    if opf == "STAY" and c.confirmed:
        p = min(c.confirmed, key=lambda q: math.hypot(q["x"] - c.rx, q["y"] - c.ry))
        from .features import standoff
        sc = standoff(c, p["x"], p["y"])
        if sc is not None:
            gx, gy = c.view.xy_of(sc[0], sc[1])
            if math.hypot(gx - c.rx, gy - c.ry) > 0.6:
                return _hold("person confirmed at %s; the mission says stay with them: moving next to them"
                             % _xy((p["x"], p["y"])), MISSION,
                             goal=(float(gx), float(gy), math.atan2(p["y"] - gy, p["x"] - gx)))
        return _hold("staying with the person at %s as ordered" % _xy((p["x"], p["y"])), MISSION)
    if M["approach_person"] and c.verify:
        for cd in c.verify:
            p, rec = cd.sighting
            if cd.eu <= 0.6:                           # standing at the standoff: this is a look
                rec["attempts"] += 1
                if rec["attempts"] >= P["verify_max_attempts"]:
                    rec["given_up"] = True
                    continue
            if not cd.feasible:
                continue
            return Decision("VERIFY", cd, None, "DWELL_SCAN",
                            "unconfirmed sighting at %s (%d hit%s); going %.1f m to look and dwell %.0f s"
                            % (_xy((p["x"], p["y"])), p["hits"], "" if p["hits"] == 1 else "s", cd.d_r,
                               P["dwell_s"]), MISSION)

    # ---- 5. go to a waypoint -------------------------------------------------------------------
    if task == "GOTO_WAYPOINT":
        return _goto_waypoint(c)

    # ---- 6. explore ----------------------------------------------------------------------------
    if task in ("SEARCH_SURVIVORS", "MAP_AREA"):
        feas = [f for f in c.explore if f.feasible]
        if feas:
            best = max(feas, key=lambda f: f.utility)
            forced = c.branch_pending and best.match and len(c.frontiers) >= 2
            sensor = "DWELL_SCAN" if (best.openness >= P["dwell_openness"] and task == "SEARCH_SURVIVORS") \
                else "MOVE"
            if best.search:
                what = "%.0f m2 of mapped floor the camera has not seen yet, %.0f m away" % (best.size_m, best.d_r)
            else:
                what = "frontier %.1f m wide, %.0f m away" % (best.size_m, best.d_r)
            if forced:
                why = "taking the first %s branch as ordered: %s" % (M["first_branch"].lower(), what)
            else:
                why = "exploring the best frontier: %s (%d feasible)" % (what, len(feas))
            if sensor == "DWELL_SCAN":
                why += "; open area, dwell %.0f s to look" % P["dwell_s"]
            return Decision("EXPLORE", best, None, sensor, why, MISSION if forced else DISCRETIONARY)

    # ---- 7. nothing feasible left --------------------------------------------------------------
    left = c.explore if task == "SEARCH_SURVIVORS" else c.frontiers
    if not left:
        why = "no frontier left to explore" + (" and the camera has seen all mapped floor"
                                                  if task == "SEARCH_SURVIVORS" else "")
    else:
        reasons = {}
        for f in left:
            for r in (f.why_not or "unreachable").split(","):
                reasons[r] = reasons.get(r, 0) + 1
        why = "%d exploration goal%s left but none feasible (%s)" % (
            len(left), "" if len(left) == 1 else "s",
            ", ".join("%s: %d" % kv for kv in sorted(reasons.items())))
    if not c.link_ok and M["on_link_loss"] == "CONTINUE_THEN_RETURN" and \
            any(f.why_not == "link silence" for f in left):
        # the rest is only out of reach because of the time allowed out of contact: check in at
        # the last contact point, then carry on (the silence clock restarts there)
        return _to_link(c, why + "; checking in before going on", MISSION)
    if M["return_when_done"]:
        return _return_or_done(c, why, DISCRETIONARY)
    if not c.link_ok and M["on_link_loss"] == "CONTINUE_THEN_RETURN":
        return _to_link(c, why + "; reporting back", DISCRETIONARY)
    return Decision("DONE", None, None, "MOVE", why + "; the mission does not ask to return", DISCRETIONARY)


def _goto_waypoint(c):
    M, P, S = c.M, c.P, c.S
    name = M["waypoint"]
    wps = P["waypoints"] or {}
    wp = wps.get(name) if name else None
    if wp is None:
        return _hold("waypoint %r is not known to the brain (pass params['waypoints'])" % (name,), MISSION)
    if isinstance(wp, dict):
        x, y = float(wp["x"]), float(wp["y"])
        yaw = math.radians(float(wp.get("yaw_deg", 0.0)))
    else:
        x, y = float(wp[0]), float(wp[1])
        yaw = math.radians(float(wp[2])) if len(wp) > 2 else 0.0
    if S.get("waypoint_reached") or math.hypot(x - c.rx, y - c.ry) <= 0.5:
        S["waypoint_reached"] = True
        if M["return_when_done"]:
            return _return_or_done(c, "arrived at waypoint %s" % name, MISSION)
        return Decision("DONE", None, None, "MOVE", "arrived at waypoint %s" % name, MISSION)
    sc = c.view.snap(x, y, 1.5, np.isfinite(c.reach))
    if sc is None:
        return _hold("waypoint %s at %s is not reachable on the known map" % (name, _xy((x, y))), MISSION)
    from .features import Cand
    n = c.graph.node(sc)
    gx, gy = c.view.xy_of(sc[0], sc[1])
    cd = Cand("EXPLORE", (float(gx), float(gy), yaw), n)
    cd.d_r = float(c.D[0][n]) * c.view.cell
    cd.d_h = float(c.D[1][n]) * c.view.cell
    if c.battery - c.energy(cd.d_r) - c.reserve(cd.d_h) - c.floor < 0:
        return _return_or_done(c, "not enough battery to reach waypoint %s and still get home" % name, SAFETY)
    d = Decision("EXPLORE", cd, None, "MOVE", "going to waypoint %s at %s (%.0f m)" % (name, _xy((x, y)), cd.d_r),
                 MISSION, extra={"target": "WAYPOINT"})
    return d
