# -*- coding: utf-8 -*-
"""Calibrate the learned policy's confidence and package the weights for the rover.

    python3 train/finalize.py train/run1/best.npz --out dhwani_brain/weights/policy.npz

WHAT "CONFIDENCE" MEANS HERE. At a discretionary decision the network picks an action (goal k
and dwell or not) with probability p = p(k) x p(dwell choice). We want P(taking the network's
action here is at least as good as taking the rules' action | p). It is measured, not assumed: on VALIDATION mines (2,000,000+; never training, never
held-out) we pause a greedy learned rollout at random discretionary decisions, copy the whole
simulation, and play it out twice under identical random numbers - once with the network's
action, once with the rules' action, the network deciding afterwards in both. The label is
1 if the network's branch scored >= the rules' branch. (When both actions agree the label is 1.)
Platt scaling  conf = sigmoid(a * logit(p) + b)  is fitted on half of these probes and its
reliability (ECE) reported on the other half. The Brain uses the network only when
conf >= learned_threshold.
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
sys.path.insert(0, HERE)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

VAL0 = 2000000
CONDS = ["nominal", "rockfall", "linkloss", "both"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y, iters=200):
    """Logistic regression of y on logit(p): two parameters, Newton's method."""
    x = logit(np.asarray(p, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        q = 1.0 / (1.0 + np.exp(-z))
        g = np.array([np.sum((q - y) * x), np.sum(q - y)]) + 1e-3 * np.array([a - 1.0, b])
        w = q * (1 - q)
        H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-3 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def ece(conf, y, bins=10):
    conf = np.asarray(conf)
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    rows = []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= 1.0)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - y[m].mean())
            rows.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": int(m.sum()),
                         "conf": round(float(conf[m].mean()), 3), "acc": round(float(y[m].mean()), 3)})
    return float(tot), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--out", default=os.path.join(ROOT, "dhwani_brain", "weights", "policy.npz"))
    ap.add_argument("--mines", type=int, default=60)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    from rollout import calib_job
    t0 = time.time()
    jobs = []
    for i in range(a.mines):
        for j, cond in enumerate(CONDS):
            jobs.append((os.path.abspath(a.weights), "drift", VAL0 + 500 + i, cond, 1000 * i + j, a.probes))
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs) as pool:
        res = pool.map(calib_job, jobs, chunksize=1)
    pairs = [pr for r in res for pr in r]
    p = np.array([x[0] for x in pairs])
    y = np.array([x[1] for x in pairs])
    differed = np.array([x[2] for x in pairs])
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(p))
    fit, test = idx[: len(idx) // 2], idx[len(idx) // 2:]
    ca, cb = fit_platt(p[fit], y[fit])
    conf_test = 1.0 / (1.0 + np.exp(-(ca * logit(p[test]) + cb)))
    raw_ece, _ = ece(p[test], y[test])
    cal_ece, rel = ece(conf_test, y[test])
    # refit on all probes for the shipped weights
    ca_all, cb_all = fit_platt(p, y)
    info = {"probes": int(len(p)), "label_rate": float(y.mean()),
            "network_differs_from_rules": int(differed.sum()),
            "network_better_or_equal_when_differs": (float(y[differed == 1].mean()) if differed.any() else None),
            "platt_half": [ca, cb],
            "platt_all": [ca_all, cb_all], "ece_raw_p": raw_ece, "ece_calibrated": cal_ece,
            "reliability_test_half": rel, "minutes": (time.time() - t0) / 60.0,
            "validation_seeds": [VAL0 + 500, VAL0 + 500 + a.mines]}
    print(json.dumps(info, indent=1))
    z = np.load(a.weights)
    d = dict((k, z[k]) for k in z.files)
    meta = json.loads(z["meta"].tobytes().decode("utf-8")) if "meta" in z.files else {}
    meta["calibration"] = info
    meta["calibration_target"] = ("P(network action (goal, dwell) scores >= rules action | p(goal) x p(dwell "
                                  "choice)), paired replays on validation mines, network continuing in both branches")
    d["calib_a"] = np.array(ca_all, dtype=np.float32)
    d["calib_b"] = np.array(cb_all, dtype=np.float32)
    d["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(a.out, **d)
    print("wrote", a.out)
    if a.log:
        with open(a.log, "w") as f:
            json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
