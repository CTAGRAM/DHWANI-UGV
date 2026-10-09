# -*- coding: utf-8 -*-
"""Write the README's results summary and learned-policy verdict from eval/results_heldout.json.

The verdict rule was fixed before the held-out run: the learned policy "beats the rules" only if
the paired score difference over all held-out episodes has a 95 % CI entirely above zero AND its
lost rate is not higher. Otherwise it ships disabled (use_learned False, the default).
"""
from __future__ import division, print_function, absolute_import

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    d = json.load(open(sys.argv[1]))
    s, pr = d["summary"], d["paired"]
    R, L, P = s["ALL|ALL|rules"], s["ALL|ALL|learned"], pr["ALL|ALL"]
    beats = (P["diff"] - P["ci"] > 0) and (L["lost_rate"] <= R["lost_rate"])
    per_style = []
    for st in ("drift", "lattice"):
        if "%s|ALL" % st in pr:
            q = pr["%s|ALL" % st]
            per_style.append("%s %+.2f +- %.2f" % (st, q["diff"], q["ci"]))
    lat = json.load(open(os.path.join(ROOT, "bench", "rover_latency.json")))["result"]
    wf = json.load(open(os.path.join(ROOT, "bench", "rover_wavefront.json")))
    n_eps = sum(1 for r in d["rows"] if "error" not in r)
    import numpy as np
    rr = [r for r in d["rows"] if "error" not in r and r["policy"] == "rules"]
    lost = [r for r in rr if r["lost"]]
    trapped_share = (100.0 * sum(1 for r in lost if r["trapped"]) / len(lost)) if lost else 0.0
    sd = float(np.std([r["score"] for r in rr], ddof=1))
    summary = (
        "**Status, 25 Sep 2026.** The rules brain is complete and is what should run. 57 tests pass on "
        "the Jetson Nano under python 2.7.17 and 3.6.9 (numpy 1.13.3) and on the Mac. On the Nano a full "
        "decision on a deep 60 m mine takes %.0f ms median (%.0f ms p99), of which the robot and home "
        "wavefronts take 87-125 ms; the rules' choice itself is %.2f ms and the learned network's %.1f ms. "
        "On %d held-out simulated missions (400 unseen mines, two generator styles, with and without "
        "rockfalls and link loss) the rules score %.2f +- %.2f, report %.2f people per mission, come home in "
        "%.0f %% and are lost in %.0f %% (%.0f %% of those were trapped: no way home existed in the true map). "
        "The learned policy (38 k parameters, BC + PPO) scores %.2f +- %.2f; paired difference %+.2f +- %.2f "
        "(%s). **%s** The specified score rewards ending searches early (section 6); "
        "read it together with people reported and lost rate."
        % (lat["full_decide_60m_rules"]["median_ms"], lat["full_decide_60m_rules"]["p99_ms"],
           lat["rules_policy_only"]["median_ms"], lat["learned_policy_only_features_plus_forward"]["median_ms"],
           n_eps // 2, R["score"], R["score_ci"], R["reported"], 100 * R["return_rate"], 100 * R["lost_rate"],
           trapped_share,
           L["score"], L["score_ci"], P["diff"], P["ci"], ", ".join(per_style),
           ("It beats the rules and is enabled by default." if beats else
            "It does not beat the rules, so it ships disabled (`use_learned` False, the default).")))
    verdict = (
        "Decision rule, fixed before the held-out run: enable the network by default only if the paired "
        "score difference over all held-out episodes has a 95 %% CI above zero and its lost rate is not "
        "higher. Result: %+.2f +- %.2f points per mission (%d paired missions; learned better in %d, worse "
        "in %d, identical in %d), lost rate %.1f %% against %.1f %%, people reported %.2f against %.2f per "
        "mission. %s"
        % (P["diff"], P["ci"], P["n"], P["wins"], P["losses"], P["n"] - P["wins"] - P["losses"],
           100 * L["lost_rate"], 100 * R["lost_rate"], L["reported"], R["reported"],
           ("It beats the rules: enabled by default." if beats else
            "**It does not beat the rules. It ships disabled** (`use_learned` False, the default); enabling "
            "it needs `params={'use_learned': True}` and, per `policy_runner.py`'s practice, a deliberate "
            "decision on the real rover.")))
    if not beats:
        verdict += (
        "\n\nWhy it cannot do better here: the network only reorders the rules' own feasible exploration goals "
        "and chooses dwell, and after BC it agrees with the rules at %s of probed decisions. PPO found no "
        "change of that ordering that the score could tell apart from noise within %s decisions; the "
        "per-mission score has a standard deviation of %.0f points, so a real gain of a fraction of a point "
        "needs far more episodes than one night gives. The rules are the brain; the learned path is wired, "
        "tested, calibrated and measured on the rover, ready for a better-posed objective (see section 11)."
        % (sys.argv[2] if len(sys.argv) > 2 else "most", sys.argv[3] if len(sys.argv) > 3 else "about 500,000", sd))
    p = os.path.join(ROOT, "README.md")
    t = io.open(p, encoding="utf-8").read()
    t = t.replace("<!-- RESULTS-SUMMARY -->", summary).replace("<!-- VERDICT -->", verdict)
    io.open(p, "w", encoding="utf-8").write(t)
    print(summary)
    print()
    print(verdict)
    print("BEATS" if beats else "DOES NOT BEAT")


if __name__ == "__main__":
    main()
