# -*- coding: utf-8 -*-
"""Turn evaluate.py results JSON(s) into the markdown tables used in README.md.

    python3 eval/report.py eval/results_heldout.json > eval/results_heldout.md
    python3 eval/report.py rules.json learned.json --merged eval/results_heldout.json   # merge runs

Several files are merged (rows concatenated, summaries and paired differences recomputed).
"""
from __future__ import division, print_function, absolute_import

import json
import sys


def pct(v):
    return "%.0f%%" % (100 * v)


def main():
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import summarise, paired
    args = sys.argv[1:]
    merged_out = None
    if "--merged" in args:
        i = args.index("--merged")
        merged_out = args[i + 1]
        args = args[:i] + args[i + 2:]
    rows, errors, sources = [], 0, []
    for path in args:
        dd = json.load(open(path))
        rows += dd["rows"]
        errors += dd.get("errors", 0)
        sources.append({"file": os.path.basename(path), "args": dd.get("args")})
    s = summarise(rows)
    pr = paired(rows)
    d = {"rows": rows, "summary": s, "paired": pr, "errors": errors, "sources": sources}
    if merged_out:
        with open(merged_out, "w") as f:
            json.dump(d, f)
    styles = sorted(set(k.split("|")[0] for k in s) - {"ALL"}) + ["ALL"]
    conds = ["nominal", "rockfall", "linkloss", "both", "ALL"]
    print("| mines | condition | policy | n | score (95% CI) | persons found | persons reported (share of all persons) "
          "| returned (95% CI) | lost (95% CI) | mission time s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for st in styles:
        for cd in conds:
            for pol in ("rules", "learned"):
                k = "%s|%s|%s" % (st, cd, pol)
                if k not in s:
                    continue
                v = s[k]
                print("| %s | %s | %s | %d | %.2f +- %.2f | %.2f +- %.2f | %.2f +- %.2f (%s) | %s (%s-%s) | %s (%s-%s) | %.0f |" % (
                    st, cd, pol, v["n"], v["score"], v["score_ci"], v["found"], v["found_ci"], v["reported"],
                    v["reported_ci"], pct(v["reported_frac"]), pct(v["return_rate"]), pct(v["return_ci"][0]),
                    pct(v["return_ci"][1]), pct(v["lost_rate"]), pct(v["lost_ci"][0]), pct(v["lost_ci"][1]),
                    v["time_s"]))
    if pr:
        print()
        print("Paired difference learned - rules (same mine, mission, world seed):")
        print()
        print("| mines | condition | n | score diff (95% CI) | learned better / worse / same |")
        print("|---|---|---|---|---|")
        order = sorted(pr, key=lambda k: (k.split("|")[0] == "ALL", k.split("|")[0], k.split("|")[1] == "ALL", k))
        for k in order:
            v = pr[k]
            st, cd = k.split("|")
            print("| %s | %s | %d | %+.2f +- %.2f | %d / %d / %d |" % (st, cd, v["n"], v["diff"], v["ci"], v["wins"],
                                                                     v["losses"], v["n"] - v["wins"] - v["losses"]))
    # share of decisions the learned policy actually made
    lr = [r for r in rows if r.get("policy") == "learned" and "error" not in r]
    if lr:
        dec = sum(r["decisions"] for r in lr)
        disc = sum(r["discretionary"] for r in lr)
        used = sum(r["learned_used"] for r in lr)
        print()
        print("Learned runs: %d decisions, %d discretionary (%.0f%%), %d taken from the network (%.0f%% of all)." % (
            dec, disc, 100.0 * disc / max(1, dec), used, 100.0 * used / max(1, dec)))
    print()
    print("errors: %d" % d.get("errors", 0))


if __name__ == "__main__":
    main()
