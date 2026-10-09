# -*- coding: utf-8 -*-
"""Decision latency of each policy (short run, < 2 min on the Nano).

    python bench/bench_latency.py [weights.npz]

1. POLICY ONLY, 1000 calls each, 6 frontier candidates: the context (map, wavefronts, frontiers)
   is built once; then the rules policy's decide(ctx) and the learned policy's features+forward
   are timed on it. This is the cost of the choice itself.
2. FULL Brain.decide on the 60 m benchmark mine (map pooling, three wavefronts, frontiers,
   camera coverage, features, policy): 25 calls per policy, because each call is ~0.1-0.2 s on
   the Nano and the rover's CPU is shared with ROS.
"""
from __future__ import division, print_function, absolute_import

import json
import math
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dhwani_brain import Brain                                              # noqa: E402
from dhwani_brain.compat import clock                                       # noqa: E402
from dhwani_brain import rules                                              # noqa: E402
from dhwani_brain.features import (build_context, candidate_slots, cand_features,   # noqa: E402
                                   global_features, K_SLOTS, NC)
from dhwani_brain.learned import LearnedPolicy, default_weights_path        # noqa: E402
from dhwani_brain.tests.synth import Mine, obs_for                          # noqa: E402


def stats(ts):
    ts = np.asarray(ts) * 1e3
    return {"median_ms": round(float(np.median(ts)), 3), "p99_ms": round(float(np.percentile(ts, 99)), 3),
            "mean_ms": round(float(np.mean(ts)), 3), "n": int(len(ts))}


def six_arm_star():
    """A hub with six radiating corridors, each revealed 6 m: exactly six frontiers."""
    m = Mine(50, 50, [], res=0.1)
    cx, cy = 25.0, 25.0
    ii, jj = np.mgrid[0:m.H, 0:m.W]
    X, Y = (jj + 0.5) * m.res, (ii + 0.5) * m.res
    free = (X - cx) ** 2 + (Y - cy) ** 2 <= 2.0 ** 2
    seen = (X - cx) ** 2 + (Y - cy) ** 2 <= 8.0 ** 2
    for k in range(6):
        a = k * math.pi / 3 + 0.2
        ux, uy = math.cos(a), math.sin(a)
        t = (X - cx) * ux + (Y - cy) * uy
        n = -(X - cx) * uy + (Y - cy) * ux
        free |= (t >= 0) & (t <= 22) & (np.abs(n) <= 0.9)
    m.free = free
    g = m.grid(None)
    g[~seen] = -1
    return m, g


def main():
    wpath = sys.argv[1] if len(sys.argv) > 1 else default_weights_path()
    out = {"python": platform.python_version(), "numpy": np.__version__, "node": platform.node(),
           "machine": platform.machine(), "weights": os.path.basename(wpath)}
    # ---- 1. policy only -------------------------------------------------------------------------
    m, g = six_arm_star()
    obs = obs_for(m, (25.0, 25.0, 0.0), home=(25.0, 25.0))
    obs["grid"] = g
    mission = {"task": "SEARCH_SURVIVORS"}
    b = Brain(mission, {"use_learned": True, "weights": wpath, "learned_threshold": 0.0})
    S = b.state
    S["start_yaw"], S["bat0"], S["odo0"] = 0.0, 90.0, 10.0
    c = build_context(obs, b.mission, b.params, S)
    n_explore = len([x for x in c.explore if x.feasible])
    out["policy_only_candidates"] = {"explore_feasible": n_explore, "frontiers": len(c.frontiers),
                                     "search_goals": len(c.searches)}
    ts = []
    rd = None
    for _ in range(1000):
        t = clock()
        rd = rules.decide(c)
        ts.append(clock() - t)
    out["rules_policy_only"] = stats(ts)
    pol = LearnedPolicy(wpath)
    out["learned_params"] = pol.n_params
    slots = candidate_slots(c)
    allowed = np.zeros(K_SLOTS, dtype=bool)
    allowed[:6] = [s is not None for s in slots[:6]]
    allowed[8] = True
    ts, ts_fwd = [], []
    for _ in range(1000):
        t = clock()
        C = np.zeros((K_SLOTS, NC), dtype=np.float32)
        present = np.zeros(K_SLOTS, dtype=bool)
        for k, cd in enumerate(slots):
            if cd is not None:
                present[k] = True
                C[k] = cand_features(c, cd, rules_pick=(cd is rd.cand))
        G = global_features(c)
        t1 = clock()
        pol.act(C, present, allowed, G)
        t2 = clock()
        ts.append(t2 - t)
        ts_fwd.append(t2 - t1)
    out["learned_policy_only_features_plus_forward"] = stats(ts)
    out["learned_forward_only"] = stats(ts_fwd)
    print(json.dumps(out))
    # ---- 2. full decide on the 60 m mine ---------------------------------------------------------
    z = np.load(os.path.join(HERE, "bench_maps.npz"))
    grid = z["drift60_grid"]
    meta = z["drift60_meta"]
    o = {"grid": grid, "res": float(meta[0]), "origin": (0.0, 0.0), "pose": (float(meta[3]), float(meta[4]), 0.0),
         "home": (float(meta[6]), float(meta[7])), "battery_pct": 90.0, "link_ok": True, "since_contact_s": 0.0,
         "elapsed_s": 600.0, "odo_m": 150.0, "persons": [], "blocked": [], "last_link_pose": None, "trail": []}
    for name, params in (("rules", None), ("learned", {"use_learned": True, "weights": wpath,
                                                        "learned_threshold": 0.0})):
        br = Brain(mission, params)
        br.decide(o)                                              # warm-up
        ts, parts = [], []
        d = None
        for _ in range(25):
            t = clock()
            d = br.decide(o)
            ts.append(clock() - t)
            parts.append(d["timing_ms"])
        out["full_decide_60m_" + name] = stats(ts)
        out["full_decide_60m_" + name]["source_last"] = d["source"]
        out["full_decide_60m_" + name]["breakdown_median_ms"] = dict(
            (k, round(float(np.median([p[k] for p in parts])), 1))
            for k in ("map", "geodesic", "frontiers", "coverage", "total"))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
