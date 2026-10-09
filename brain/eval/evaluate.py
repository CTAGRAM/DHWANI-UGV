# -*- coding: utf-8 -*-
"""Batch evaluation in the simulator: policies x conditions x mines, in parallel.

    python3 eval/evaluate.py --styles drift --seeds 9000000:9000200 --policies rules learned \
        --conds all --procs 10 --out eval/results_heldout.json

Seed ranges (never mix them up):
    training     1000000 - 1999999   (drift only)
    validation   2000000 - 2099999   (drift only; calibration, threshold, checkpoint choice)
    held-out     9000000 - 9099999   drift, unseen seeds
                 9500000 - 9599999   lattice: a generator style never used in training
Both policies run on exactly the same mines, missions and world seeds, so differences are paired.
"""
from __future__ import division, print_function, absolute_import

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CONDS = {"nominal": (False, False), "rockfall": (True, False), "linkloss": (False, True),
         "both": (True, True)}


def policy_params(name, weights=None, threshold=None):
    if name == "rules":
        return None
    if name == "learned":
        p = {"use_learned": True}
        if weights:
            p["weights"] = weights
        if threshold is not None:
            p["learned_threshold"] = threshold
        return p
    raise ValueError(name)


def _job(a):
    from dhwani_brain.sim.runner import run_episode
    style, seed, cond, pol, weights, thr = a
    rf, ll = CONDS[cond]
    t = time.time()
    try:
        m = run_episode(style, seed, rockfall=rf, link_loss=ll, params=policy_params(pol, weights, thr))
    except Exception as e:                                   # keep the batch alive, record it
        import traceback
        return {"style": style, "seed": seed, "cond": cond, "policy": pol, "error": traceback.format_exc()}
    m["cond"] = cond
    m["policy"] = pol
    m["wall_s"] = time.time() - t
    return m


def run_jobs(jobs, procs):
    if procs <= 1:
        return [_job(j) for j in jobs]
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    with ctx.Pool(procs) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            out.append(r)
            if (i + 1) % 50 == 0:
                print("  %d/%d episodes" % (i + 1, len(jobs)), flush=True)
        return out


def mean_ci(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if n < 2:
        return m, float("nan")
    se = float(x.std(ddof=1)) / math.sqrt(n)
    return m, 1.96 * se


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def summarise(rows):
    """Per (style, cond, policy): mean score +- CI, persons reported, return / lost rates."""
    out = {}
    ok = [r for r in rows if "error" not in r]
    keys = sorted(set((r["style"], r["cond"], r["policy"]) for r in ok))
    keys += sorted(set((r["style"], "ALL", r["policy"]) for r in ok))
    keys += sorted(set(("ALL", "ALL", r["policy"]) for r in ok))
    for key in keys:
        rs = [r for r in ok if (key[0] == "ALL" or r["style"] == key[0]) and (key[1] == "ALL" or r["cond"] == key[1])
              and r["policy"] == key[2]]
        n = len(rs)
        s, sci = mean_ci([r["score"] for r in rs])
        rep, rci = mean_ci([r["reported"] for r in rs])
        found, fci = mean_ci([r["found"] for r in rs])
        tt, tci = mean_ci([r["time_s"] for r in rs])
        persons = sum(r["persons"] for r in rs)
        ret = wilson(sum(r["returned"] for r in rs), n)
        lost = wilson(sum(r["lost"] for r in rs), n)
        out["|".join(key)] = {"n": n, "score": s, "score_ci": sci, "reported": rep, "reported_ci": rci,
                              "found": found, "found_ci": fci,
                              "reported_frac": sum(r["reported"] for r in rs) / max(1, persons),
                              "return_rate": ret[0], "return_ci": [ret[1], ret[2]],
                              "lost_rate": lost[0], "lost_ci": [lost[1], lost[2]],
                              "time_s": tt, "time_ci": tci,
                              "learned_share": sum(r["learned_used"] for r in rs) / max(1, sum(r["decisions"] for r in rs))}
    return out


def paired(rows, a="learned", b="rules"):
    """Paired score difference a - b over identical (style, seed, cond)."""
    idx = {}
    for r in rows:
        if "error" in r:
            continue
        idx[(r["style"], r["seed"], r["cond"], r["policy"])] = r["score"]
    out = {}
    for style in sorted(set(k[0] for k in idx)) + ["ALL"]:
        for cond in sorted(set(k[2] for k in idx)) + ["ALL"]:
            if style == "ALL" and cond != "ALL":
                continue
            diffs = []
            for (st, sd, cd, pol), v in idx.items():
                if (style != "ALL" and st != style) or pol != a or (cond != "ALL" and cd != cond):
                    continue
                w = idx.get((st, sd, cd, b))
                if w is not None:
                    diffs.append(v - w)
            if diffs:
                m, ci = mean_ci(diffs)
                out["%s|%s" % (style, cond)] = {"n": len(diffs), "diff": m, "ci": ci,
                                                "wins": int(np.sum(np.array(diffs) > 1e-9)),
                                                "losses": int(np.sum(np.array(diffs) < -1e-9))}
    return out


def parse_range(s):
    a, b = s.split(":")
    return list(range(int(a), int(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", nargs="+", default=["drift"])
    ap.add_argument("--seeds", default="9000000:9000010", help="a:b for drift; lattice uses a+500000")
    ap.add_argument("--lattice-seeds", default=None)
    ap.add_argument("--conds", nargs="+", default=["all"])
    ap.add_argument("--policies", nargs="+", default=["rules"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    conds = list(CONDS) if a.conds == ["all"] else a.conds
    jobs = []
    for style in a.styles:
        if style == "lattice":
            seeds = parse_range(a.lattice_seeds) if a.lattice_seeds else [s + 500000 for s in parse_range(a.seeds)]
        else:
            seeds = parse_range(a.seeds)
        for seed in seeds:
            for cond in conds:
                for pol in a.policies:
                    jobs.append((style, seed, cond, pol, a.weights, a.threshold))
    print("%d episodes on %d processes" % (len(jobs), a.procs), flush=True)
    t = time.time()
    rows = run_jobs(jobs, a.procs)
    print("done in %.0f s" % (time.time() - t))
    errs = [r for r in rows if "error" in r]
    for e in errs[:5]:
        print("ERROR", e["style"], e["seed"], e["cond"], e["policy"], e["error"][-800:])
    summ = summarise(rows)
    for k, v in sorted(summ.items()):
        print("%-28s n=%3d score %6.2f +- %5.2f | found %.2f reported %.2f (%.0f%% of persons) | return %.0f%% "
              "lost %.1f%% | %5.0f s | learned %.0f%%" % (
                  k, v["n"], v["score"], v["score_ci"], v["found"], v["reported"], 100 * v["reported_frac"],
                  100 * v["return_rate"], 100 * v["lost_rate"], v["time_s"], 100 * v["learned_share"]))
    pr = paired(rows) if "learned" in a.policies and "rules" in a.policies else {}
    for k, v in sorted(pr.items()):
        print("paired learned-rules %-20s n=%3d diff %+.2f +- %.2f (wins %d, losses %d)" % (
            k, v["n"], v["diff"], v["ci"], v["wins"], v["losses"]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "rows": [dict((k, v) for k, v in r.items() if k != "trace") for r in rows],
                       "summary": summ, "paired": pr, "errors": len(errs)}, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
