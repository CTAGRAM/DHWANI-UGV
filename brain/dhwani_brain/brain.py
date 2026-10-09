# -*- coding: utf-8 -*-
"""Brain: the public entry point.

    from dhwani_brain import Brain
    brain = Brain(mission_dict, params=None)
    decision = brain.decide(obs)

One Brain per mission. It keeps a little memory between calls (heading and battery at the start,
whether the first branch was taken, verify attempts per sighting, the previous goal, returns
latched by the mission rules); everything else is recomputed from obs on every call.

ARBITRATION. The rules policy always runs. Its decision is final when its tier is 'safety' or
'mission', and whenever it decides the search is over. When the rules are EXPLOREing (tier
'discretionary') and params['use_learned'] is true, the learned policy picks which of the same
feasible exploration goals to take and whether to dwell there; its pick is used only if its
calibrated confidence >= params['learned_threshold'] and the candidate passes the safety
re-check. Otherwise the rules decision stands. The decision says which one decided ('source').
"""
from __future__ import division, print_function, absolute_import

import math
import traceback

import numpy as np

from .compat import clock, fnum
from .params import normalize_mission, merge_params
from .features import (build_context, candidate_slots, cand_features, global_features,
                       K_SLOTS, NC, KINDS)
from .geodesic import descend
from . import rules
from .rules import Decision, DISCRETIONARY


class Brain(object):
    def __init__(self, mission, params=None):
        self.mission = normalize_mission(mission)
        self.params = merge_params(params)
        self.state = {}
        if self.params["start_yaw"] is not None:
            self.state["start_yaw"] = float(self.params["start_yaw"])
        self.policy = None
        self.policy_error = None
        if self.params["use_learned"]:
            try:
                from .learned import LearnedPolicy
                self.policy = LearnedPolicy(self.params["weights"])
            except Exception as e:           # a missing/corrupt weight file must not stop the rover
                self.policy_error = "%s: %s" % (type(e).__name__, e)
        self.calls = 0
        self.last = None

    # ------------------------------------------------------------------------------------------
    def decide(self, obs, sample=None):
        """obs: dict (see README). sample: optional callable used by the trainer to replace the
        learned policy's argmax with a sampled action; never set it on the rover."""
        t0 = clock()
        S = self.state
        self.calls += 1
        try:
            pose = obs["pose"]
            if "start_yaw" not in S:
                S["start_yaw"] = float(pose[2]) if len(pose) > 2 and pose[2] is not None else 0.0
            if "bat0" not in S:
                S["bat0"] = fnum(obs.get("battery_pct"), 100.0)
                S["odo0"] = fnum(obs.get("odo_m"), 0.0)
            c = build_context(obs, self.mission, self.params, S)
            rd = rules.decide(c)
            slots = candidate_slots(c)
            C = np.zeros((K_SLOTS, NC), dtype=np.float32)
            present = np.zeros(K_SLOTS, dtype=bool)
            for k, cd in enumerate(slots):
                if cd is None:
                    continue
                present[k] = True
                C[k] = cand_features(c, cd, rules_pick=(cd is rd.cand))
                cd.feat = C[k]
            G = global_features(c)
            allowed = self._allowed(c, slots, rd)
            final, linfo = rd, None
            if rd.tier == DISCRETIONARY and allowed.any() and (self.policy is not None or sample is not None):
                final, linfo = self._learned(c, rd, slots, C, present, allowed, G, sample)
            self._remember(c, final)
            out = self._format(c, final, rd, slots, linfo)
            out["timing_ms"] = {"map": round(c.t_map_ms, 1), "geodesic": round(c.t_geo_ms, 1),
                                "frontiers": round(c.t_fr_ms, 1), "coverage": round(c.t_cov_ms, 1),
                                "total": round((clock() - t0) * 1e3, 1)}
            # training hook: the arrays the network saw (not part of the rover contract)
            self.last = {"C": C, "present": present, "allowed": allowed, "G": G, "tier": rd.tier,
                         "rules_slot": self._slot_of(slots, rd.cand), "rules_dwell": rd.sensor == "DWELL_SCAN",
                         "learned": linfo, "ctx": c}
            return out
        except Exception as e:
            self.last = None
            return {"kind": "HOLD", "goal": None, "sensor": "MOVE", "source": "rules", "confidence": 1.0,
                    "reason": "brain error, holding: %s: %s" % (type(e).__name__, e), "tier": "safety",
                    "candidates": [], "path": [], "error": traceback.format_exc(),
                    "timing_ms": {"total": round((clock() - t0) * 1e3, 1)}}

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _slot_of(slots, cand):
        for k, cd in enumerate(slots):
            if cd is not None and cd is cand:
                return k
        return -1

    def _allowed(self, c, slots, rd):
        """What the learned policy may pick when the rules call is discretionary: one of the
        feasible exploration goals (and whether to dwell there). Ending the search is NOT on
        the list: under the task's score (0.01 point per second) going home at once beats a
        full search on average, so a network trained on that score learns to abandon searches.
        When the search is over is mission semantics, and stays with the rules."""
        a = np.zeros(K_SLOTS, dtype=bool)
        if rd.tier != DISCRETIONARY or rd.kind != "EXPLORE":
            return a
        if rd.cand is None or rd.cand not in slots[:6]:        # e.g. a waypoint: never hand it over
            return a
        for k in range(6):
            cd = slots[k]
            a[k] = cd is not None and cd.feasible and np.isfinite(cd.d_r)
        return a

    def _learned(self, c, rd, slots, C, present, allowed, G, sample):
        P = self.params
        if sample is not None:
            k, dwell, conf, probs, pd = sample(C, present, allowed, G)
        else:
            try:
                k, dwell, conf, probs, pd = self.policy.act(C, present, allowed, G)
            except Exception as e:
                return rd, {"error": "%s: %s" % (type(e).__name__, e)}
        info = {"slot": int(k), "confidence": float(conf), "dwell": bool(dwell),
                "probs": [round(float(p), 4) for p in probs]}
        if sample is None and conf < P["learned_threshold"]:
            info["used"] = False
            info["why"] = "confidence %.2f < threshold %.2f" % (conf, P["learned_threshold"])
            return rd, info
        cd = slots[k] if 0 <= k < len(slots) else None
        sensor = "DWELL_SCAN" if dwell else "MOVE"
        if cd is None or not allowed[k] or not (cd.kind == "EXPLORE" and cd.feasible and np.isfinite(cd.d_r)):
            info["used"] = False
            info["why"] = "pick failed the safety re-check"
            return rd, info
        d = Decision("EXPLORE", cd, None, sensor, "", DISCRETIONARY)
        if cd.search:
            what = "looking at %.0f m2 of floor the camera has not seen, %.0f m away" % (cd.size_m, cd.d_r)
        else:
            what = "exploring frontier %.1f m wide, %.0f m away" % (cd.size_m, cd.d_r)
        if sensor == "DWELL_SCAN":
            what += ", dwell %.0f s" % P["dwell_s"]
        same = (d.cand is rd.cand and d.kind == rd.kind and d.sensor == rd.sensor)
        d.reason = "learned policy (confidence %.2f): %s%s" % (
            conf, what, "" if same else "; rules would have: " + rd.reason)
        d.source = "learned"
        d.confidence = float(conf)
        info["used"] = True
        return d, info

    def _remember(self, c, d):
        S = self.state
        S["prev_kind"] = d.kind
        if d.goal is not None:
            S["prev_goal"] = (d.goal[0], d.goal[1])
        if d.kind == "EXPLORE" and d.cand is not None and c.branch_pending and d.cand.match \
                and len(c.frontiers) >= 2:
            S["branch_committed"] = True
            S["branch_taken_at"] = (c.rx, c.ry)

    def _path(self, c, d):
        cd = d.cand
        if d.goal is None:
            return []
        n = cd.node if cd is not None and cd.node is not None and cd.node >= 0 else \
            c.graph.node(c.view.cell_of(d.goal[0], d.goal[1]))
        if n < 0:
            return []
        nodes = descend(c.graph, c.D[0], n)
        if not nodes:
            return []
        nodes = nodes[::-1]
        step = max(1, int(round(0.4 / c.view.cell)))
        keep = nodes[::step]
        if keep[-1] != nodes[-1]:
            keep.append(nodes[-1])
        xs, ys = c.view.xy_of(c.graph.rr[keep], c.graph.cc[keep])
        return [(round(float(x), 2), round(float(y), 2)) for x, y in zip(xs, ys)]

    def _format(self, c, d, rd, slots, linfo):
        goal = None
        if d.goal is not None:
            goal = (round(float(d.goal[0]), 3), round(float(d.goal[1]), 3), round(float(d.goal[2]), 3))
        cands = []
        probs = linfo.get("probs") if linfo else None
        for k, cd in enumerate(slots):
            if cd is None:
                continue
            e = {"slot": k, "kind": cd.kind,
                 "goal": None if cd.goal is None else (round(cd.goal[0], 3), round(cd.goal[1], 3),
                                                        round(cd.goal[2], 3)),
                 "score": round(float(cd.utility), 4) if cd.kind == "EXPLORE" else
                 (1.0 if cd is rd.cand else 0.0),
                 "feasible": bool(cd.feasible), "why_not": cd.why_not,
                 "d_robot_m": round(cd.d_r, 2) if np.isfinite(cd.d_r) else None,
                 "d_home_m": round(cd.d_h, 2) if np.isfinite(cd.d_h) else None,
                 "chosen": cd is d.cand,
                 "features": [round(float(v), 4) for v in (cd.feat if cd.feat is not None else [])]}
            if probs is not None:
                e["p_learned"] = probs[k]
            cands.append(e)
        out = {"kind": d.kind, "goal": goal, "sensor": d.sensor, "reason": d.reason,
               "source": d.source, "confidence": round(float(d.confidence), 4), "tier": d.tier,
               "candidates": cands, "path": self._path(c, d)}
        if d.source != "rules":
            out["rules_decision"] = {"kind": rd.kind, "reason": rd.reason,
                                     "goal": None if rd.goal is None else tuple(round(float(v), 3) for v in rd.goal)}
        if linfo is not None:
            out["learned"] = dict((k, v) for k, v in linfo.items() if k != "probs")
        if self.policy_error:
            out["learned_error"] = self.policy_error
        out["status"] = {
            "battery_pct": round(c.battery, 1), "floor_pct": round(c.floor, 1),
            "reserve_pct": round(c.reserve_now, 1) if np.isfinite(c.reserve_now) else None,
            "d_home_m": round(c.d_home_now, 1) if np.isfinite(c.d_home_now) else None,
            "home_reachable": c.home_reachable, "at_home": bool(c.at_home),
            "rate_pct_per_m": round(c.rate, 4), "frontiers": c.n_frontiers_all,
            "branch_pending": c.branch_pending, "latched": self.state.get("latched"),
            "grid_cells": [int(c.view.shape[0]), int(c.view.shape[1])], "graph_nodes": int(c.graph.N)}
        out.update(d.extra)
        return out
