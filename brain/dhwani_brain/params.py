# -*- coding: utf-8 -*-
"""Default parameters and mission normalisation.

Mission defaults follow docs/INTERFACES.md section 1 exactly; anything the parser left
null or omitted is filled here, so the rest of the brain never has to ask "what if it is None".
"""
from __future__ import division, print_function, absolute_import

import copy

from .compat import upper, fnum

TASKS = ("SEARCH_SURVIVORS", "MAP_AREA", "GOTO_WAYPOINT", "RETURN_HOME", "HALT", "UNSUPPORTED")
PERSON_OPTS = ("REPORT_AND_CONTINUE", "REPORT_AND_RETURN", "STAY")
LINK_OPTS = ("CONTINUE_THEN_RETURN", "RETURN_NOW", "CONTINUE", "HOLD")
BRANCH_OPTS = ("ANY", "LEFT", "RIGHT", "STRAIGHT")

DEFAULT_PARAMS = {
    # planning grid
    "cell_m": 0.2,              # coarse planning cell (the rover's gmapping map is 0.05 m)
    "inflate_m": 0.15,          # obstacles and blocked regions grow by this much
    "occ_thresh": 65,           # ROS occupancy >= this is an obstacle (map_server's 0.65)
    "free_thresh": 25,          # 0..this is free; in between counts as unknown
    "robot_snap_m": 1.0,        # robot may sit inside the inflation band; snap to free space
    "escape_m": 1.0,            # ...and may leave the band within this radius (see MapView.open_escape)
    "goal_snap_m": 1.5,         # frontier centroid -> nearest reachable cell within this
    "min_frontier_cells": 3,    # smaller frontier clusters are map noise (3 cells = 0.6 m)
    "frontier_split_m": 3.0,    # longer frontiers are split into pieces this long
    # motion and energy model (tune drain_* to the real pack; the brain also learns online)
    "speed_mps": 0.30,          # NORMAL ground speed used for time estimates
    "slow_speed_mps": 0.18,     # SLOW
    "drain_pct_per_m": 0.04,    # battery percent per metre driven
    "drain_pct_per_s": 0.012,   # battery percent per second, moving or not
    "path_factor": 1.15,        # real driven path / geodesic length (TEB swerves, Ackermann)
    "reserve_frac": 0.30,       # return reserve = energy home x (1 + reserve_frac) + reserve_pct
    "reserve_pct": 4.0,
    "battery_floor_pct": 5.0,   # never plan to arrive below this even without min_battery_pct
    "time_margin_frac": 0.20,   # same idea for max_duration_min / max_distance_m
    "redecide_s": 10.0,         # the executive re-asks at least this often; used as lookahead
    "online_rate_min_m": 20.0,  # after this much odometry, trust the measured drain if worse
    # candidates (the learned policy's slot layout is fixed: 6 exploration goals, 2 verify)
    "k_verify": 2,
    "dwell_s": 5.0,
    "home_radius_m": 1.0,
    "verify_standoff_m": 1.5,
    "verify_max_attempts": 2,   # dwell-scans at a sighting before it is written off as false
    "link_radius_m": 1.0,       # "at the last-link point"
    # mission semantics
    "link_grace_s": 600.0,      # CONTINUE_THEN_RETURN: silence allowed before checking in
    "route_tortuosity": 1.5,    # unknown route home estimate = straight line x this
    "first_branch_gain": 4.0,
    "persist_gain": 1.3,        # hysteresis toward the previous exploration goal
    "dwell_openness": 0.55,     # rules dwell at frontier goals this open (chamber-like)
    "cam_cover_m": 3.0,         # floor within this geodesic distance of the trail counts as seen
    "search_min_area_m2": 2.0,  # smaller unseen patches are not worth a trip
    "search_gain": 0.35,        # rules utility per unseen m2 (frontier gain is ~2-5)
    # learned policy
    "use_learned": False,
    "learned_threshold": 0.5,   # on the CALIBRATED scale: P(network pick >= rules pick) (README 5.5)
    "weights": None,            # path to .npz; None = the package's weights/policy.npz
    # context supplied by the executive
    "waypoints": None,          # {"A": {"x":..,"y":..,"yaw_deg":..}} for GOTO_WAYPOINT
    "start_yaw": None,          # heading at mission start (rad); default = pose at first decide()
}

_TASK_DEFAULTS = {
    "SEARCH_SURVIVORS": dict(report_person=True, approach_person=True,
                             on_person_found="REPORT_AND_CONTINUE",
                             on_link_loss="CONTINUE_THEN_RETURN", return_when_done=True),
    "MAP_AREA": dict(report_person=True, approach_person=False,
                     on_person_found="REPORT_AND_CONTINUE",
                     on_link_loss="CONTINUE_THEN_RETURN", return_when_done=True),
    "GOTO_WAYPOINT": dict(report_person=True, approach_person=False,
                          on_person_found="REPORT_AND_CONTINUE",
                          on_link_loss="RETURN_NOW", return_when_done=False),
    "RETURN_HOME": dict(report_person=True, approach_person=False,
                        on_person_found="REPORT_AND_CONTINUE",
                        on_link_loss="CONTINUE", return_when_done=False),
    "HALT": dict(report_person=True, approach_person=False,
                 on_person_found="REPORT_AND_CONTINUE",
                 on_link_loss="HOLD", return_when_done=False),
    "UNSUPPORTED": dict(report_person=True, approach_person=False,
                        on_person_found="REPORT_AND_CONTINUE",
                        on_link_loss="HOLD", return_when_done=False),
}


def _bool(x, default):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = upper(x, "")
    if s in ("TRUE", "YES", "1"):
        return True
    if s in ("FALSE", "NO", "0"):
        return False
    return bool(x)


def normalize_mission(m):
    """Return a complete mission dict with every field valid; unknown enums fall back safely."""
    m = dict(m or {})
    task = upper(m.get("task"), "UNSUPPORTED")
    if task not in TASKS:
        task = "UNSUPPORTED"
    d = _TASK_DEFAULTS[task]
    out = {"task": task}
    out["report_person"] = _bool(m.get("report_person"), d["report_person"])
    out["approach_person"] = _bool(m.get("approach_person"), d["approach_person"])
    opf = upper(m.get("on_person_found"), d["on_person_found"])
    out["on_person_found"] = opf if opf in PERSON_OPTS else d["on_person_found"]
    oll = upper(m.get("on_link_loss"), d["on_link_loss"])
    out["on_link_loss"] = oll if oll in LINK_OPTS else d["on_link_loss"]
    out["return_when_done"] = _bool(m.get("return_when_done"), d["return_when_done"])
    fb = upper(m.get("first_branch"), "ANY")
    out["first_branch"] = fb if fb in BRANCH_OPTS else "ANY"
    sp = upper(m.get("speed"), "NORMAL")
    out["speed"] = sp if sp in ("NORMAL", "SLOW") else "NORMAL"
    wp = m.get("waypoint")
    out["waypoint"] = upper(wp, None) if wp is not None else None
    dur = fnum(m.get("max_duration_min"))
    dist = fnum(m.get("max_distance_m"))
    bat = fnum(m.get("min_battery_pct"))
    out["max_duration_min"] = dur if dur is not None and dur > 0 else None
    out["max_distance_m"] = dist if dist is not None and dist > 0 else None
    out["min_battery_pct"] = min(max(bat, 0.0), 95.0) if bat is not None else None
    out["confidence"] = fnum(m.get("confidence"), 1.0)
    out["understood"] = _bool(m.get("understood"), task != "UNSUPPORTED")
    out["text"] = m.get("text", "")
    return out


def merge_params(params):
    p = copy.deepcopy(DEFAULT_PARAMS)
    for k, v in (params or {}).items():
        if k not in p:
            raise KeyError("unknown brain parameter %r (known: %s)" % (k, ", ".join(sorted(p))))
        p[k] = v
    return p
