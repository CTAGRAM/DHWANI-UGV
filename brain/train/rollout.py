# -*- coding: utf-8 -*-
"""Rollout workers for training (NumPy only - no torch in the worker processes).

Workers run the simulator with the SAME inference code the rover uses (dhwani_brain.learned),
so what is trained is exactly what ships. Three jobs:
  bc_job      rules policy; records (features, rules choice) at every discretionary decision
  ppo_job     learned policy SAMPLING its choice at discretionary decisions; records features,
              action, log-probability and the score accrued until the next such decision
  eval_job    one episode with a given policy (greedy learned or rules); returns metrics
"""
from __future__ import division, print_function, absolute_import

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                                   # noqa: E402

from dhwani_brain.sim.runner import Episode                          # noqa: E402
from dhwani_brain.learned import LearnedPolicy                       # noqa: E402

CONDS = {"nominal": (False, False), "rockfall": (True, False), "linkloss": (False, True),
         "both": (True, True)}
END_SLOT = 8


def _pack(steps):
    if not steps:
        return None
    out = {}
    for key in ("C", "present", "allowed", "G"):
        out[key] = np.stack([s[key] for s in steps])
    for key in ("k", "dw", "logp", "r", "explore"):
        if key in steps[0]:
            out[key] = np.array([s[key] for s in steps])
    return out


def bc_job(a):
    style, seed, cond = a
    rf, ll = CONDS[cond]
    ep = Episode(style, seed, rf, ll)
    steps = []

    def rec(last, d, world):
        if last is None or last["tier"] != "discretionary" or not last["allowed"].any():
            return
        k = last["rules_slot"]
        if k < 0 or k >= 8:
            k = END_SLOT
        if not last["allowed"][k]:
            return
        steps.append({"C": last["C"], "present": last["present"], "allowed": last["allowed"], "G": last["G"],
                      "k": k, "dw": d["sensor"] == "DWELL_SCAN", "explore": k < 6})
    ep.run(recorder=rec)
    return _pack(steps), ep.metrics()


def ppo_job(a):
    wpath, style, seed, cond, sseed = a
    rf, ll = CONDS[cond]
    pol = LearnedPolicy(wpath)
    rng = np.random.RandomState(sseed)
    ep = Episode(style, seed, rf, ll)
    steps = []

    def sampler(C, present, allowed, G):
        probs, dwell = pol.forward(C, present, allowed, G)
        p = probs.astype(np.float64)
        p = p / p.sum()
        k = int(rng.choice(len(p), p=p))
        explore = k < 6
        dw = bool(rng.rand() < dwell[k]) if explore else False
        lp = math.log(max(p[k], 1e-12))
        if explore:
            lp += math.log(max(float(dwell[k]) if dw else 1.0 - float(dwell[k]), 1e-12))
        steps.append({"C": C.copy(), "present": present.copy(), "allowed": allowed.copy(), "G": G.copy(),
                      "k": k, "dw": dw, "logp": lp, "explore": explore, "s": ep.score_now()})
        return k, dw, float(p[k]), probs, dwell

    ep.run(sampler=sampler)
    m = ep.metrics()
    for i, s in enumerate(steps):
        nxt = steps[i + 1]["s"] if i + 1 < len(steps) else m["score"]
        s["r"] = nxt - s["s"]
    return _pack(steps), m


def eval_job(a):
    wpath, style, seed, cond, thr = a
    rf, ll = CONDS[cond]
    params = None
    if wpath:
        params = {"use_learned": True, "weights": wpath, "learned_threshold": thr}
    ep = Episode(style, seed, rf, ll, params=params)
    ep.run()
    m = ep.metrics()
    m["cond"] = cond
    m["policy"] = "learned" if wpath else "rules"
    return m


def calib_job(a):
    """Paired one-step comparison at randomly chosen discretionary decisions of a greedy learned
    rollout: take the network's action (goal and dwell) vs the rules' action, then continue with
    the network in both branches under identical random numbers.
    Returns [(p_joint_of_network_action, better_or_equal, differed)]."""
    import copy
    wpath, style, seed, cond, sseed, n_probe = a
    rf, ll = CONDS[cond]
    pol = LearnedPolicy(wpath)
    rng = np.random.RandomState(sseed)
    params = {"use_learned": True, "weights": wpath, "learned_threshold": 0.0}
    ep = Episode(style, seed, rf, ll, params=params)
    out = []
    probes = set(rng.choice(np.arange(1, 400), size=n_probe, replace=False).tolist())
    counter = {"n": 0}

    def stop(e):
        counter["n"] += 1
        return counter["n"] in probes

    while not ep.run(stop=stop):
        snap = copy.deepcopy(ep)
        peek = copy.deepcopy(snap)
        seen = {}

        def grab(C, present, allowed, G):
            probs, dwell = pol.forward(C, present, allowed, G)
            k = int(np.argmax(probs))
            dw = bool(dwell[k] > 0.5)
            seen.update(k=k, dw=dw, p=float(probs[k]) * (float(dwell[k]) if dw else 1.0 - float(dwell[k])))
            return k, dw, float(probs[k]), probs, dwell
        peek.brain.policy = None
        peek.brain.decide(peek.world.obs(), sample=grab)
        last = peek.brain.last
        if not seen or last is None or last["tier"] != "discretionary":
            continue
        net_act = (seen["k"], seen["dw"])
        rules_act = (last["rules_slot"], bool(last["rules_dwell"]))
        if net_act == rules_act:
            out.append((seen["p"], 1, 0))
            continue
        scores = []
        for act in (net_act, rules_act):
            br = copy.deepcopy(snap)
            forced = {"done": False}

            def force(C, present, allowed, G, act=act, forced=forced):
                probs, dwell = pol.forward(C, present, allowed, G)
                if not forced["done"]:
                    forced["done"] = True
                    k, dw = act
                    if not allowed[k]:
                        k = int(np.argmax(probs))
                else:
                    k = int(np.argmax(probs))
                    dw = bool(dwell[k] > 0.5)
                return k, dw, float(probs[k]), probs, dwell
            br.brain.policy = None
            br.run(sampler=force)
            scores.append(br.metrics()["score"])
        out.append((seen["p"], int(scores[0] >= scores[1] - 1e-9), 1))
    return out
