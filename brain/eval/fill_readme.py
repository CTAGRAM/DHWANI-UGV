# -*- coding: utf-8 -*-
"""Fill README.md's evaluation section from eval/results_heldout.json (no hand-copied numbers).

    python3 eval/fill_readme.py eval/results_heldout.json
"""
from __future__ import division, print_function, absolute_import

import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    path = sys.argv[1]
    md = subprocess.check_output([sys.executable, os.path.join(HERE, "report.py"), path]).decode("utf-8")
    d = json.load(open(path))
    s = d["summary"]
    n_mines = {}
    for r in d["rows"]:
        if "error" not in r:
            n_mines.setdefault(r["style"], set()).add(r["seed"])
    head = ("Held-out mines, never used for training, validation, calibration or any choice of parameters: "
            "%d unseen **drift** mines (seeds 9,000,000+) and %d **lattice** mines (seeds 9,500,000+, a "
            "generator style never used in training). Each mine is run in four conditions (nominal, "
            "rockfall, link loss, both) with the same mission, the same world random numbers and both "
            "policies, so differences are paired. Mission mix: `sim/runner.py` `sample_mission` (80 %% "
            "SEARCH_SURVIVORS, 20 %% MAP_AREA, all on_link_loss / on_person_found / first_branch options, "
            "some time, distance and battery limits). 95 %% CIs: mean +- 1.96 standard errors; rates: "
            "Wilson intervals. Device: Mac M4 (the simulator), %d episodes, %d errors. Every episode is in "
            "`eval/results_heldout.json`.\n\n" % (len(n_mines.get("drift", ())), len(n_mines.get("lattice", ())),
                                                   len(d["rows"]), d.get("errors", 0)))
    readme = io.open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    if "<!-- EVAL -->" in readme:
        readme = readme.replace("<!-- EVAL -->", head + md)
    rules_all = s.get("ALL|ALL|rules")
    if rules_all and "<!-- RULES-AVG -->" in readme:
        readme = readme.replace("<!-- RULES-AVG -->", "%.1f" % rules_all["score"])
    io.open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write(readme)
    print("filled; rules ALL score %.2f" % rules_all["score"] if rules_all else "filled")


if __name__ == "__main__":
    main()
