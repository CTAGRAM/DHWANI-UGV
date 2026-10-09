# -*- coding: utf-8 -*-
"""Run one simulated mission with the rules policy and save a PNG of the map, path and decisions.

    python3 examples/run_mission.py                       # default: drift mine, seed 4247, rockfall + link loss
    python3 examples/run_mission.py --style lattice --seed 9500003 --no-rockfall --out mission.png
    python3 examples/run_mission.py --learned             # use the learned policy where it may decide

Needs matplotlib (Mac only; the brain itself does not).
"""
from __future__ import division, print_function, absolute_import

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dhwani_brain.sim.runner import run_episode           # noqa: E402
from dhwani_brain.sim.mapgen import generate              # noqa: E402

KIND_COLOUR = {"EXPLORE": "#1f77b4", "VERIFY": "#9467bd", "RETURN_HOME": "#2ca02c",
               "RETURN_TO_LINK": "#ff7f0e", "HOLD": "#d62728", "DONE": "#000000"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="drift")
    ap.add_argument("--seed", type=int, default=4247)
    ap.add_argument("--no-rockfall", action="store_true")
    ap.add_argument("--no-link-loss", action="store_true")
    ap.add_argument("--learned", action="store_true")
    ap.add_argument("--mission", default=None, help="JSON mission dict (default: sampled from the seed)")
    ap.add_argument("--out", default=os.path.join(HERE, "mission.png"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = {"use_learned": True, "learned_threshold": 0.0} if a.learned else None
    mission = json.loads(a.mission) if a.mission else None
    mine = generate(a.style, a.seed)
    m = run_episode(a.style, a.seed, rockfall=not a.no_rockfall, link_loss=not a.no_link_loss,
                    params=params, trace=True, mission=mission, mine=mine)
    w = m["world"]
    tr = m["trace"]
    res = mine.res
    H, W = mine.free.shape
    ext = (0, W * res, 0, H * res)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.07, 0.30, 0.9, 0.64])
    axn = fig.add_axes([0.02, 0.01, 0.96, 0.24])
    axn.axis("off")
    img = np.ones((H, W, 3))
    img[mine.free] = (0.93, 0.93, 0.93)                      # truth, faint
    img[w.known == 0] = (0.72, 0.84, 0.95)                    # seen free
    img[w.known == 100] = (0.15, 0.15, 0.15)                  # seen walls
    img[mine.free & ~w.truth] = (0.55, 0.35, 0.2)             # rubble
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    xy = np.array([(x, y) for (_, x, y, _) in tr["xy"]]) if tr["xy"] else np.zeros((0, 2))
    link = np.array([ok for (_, _, _, ok) in tr["xy"]], dtype=bool) if tr["xy"] else np.zeros(0, bool)
    if len(xy):
        ax.plot(xy[:, 0], xy[:, 1], "-", color="#444444", lw=1.0, alpha=0.6, label="driven path")
        ax.scatter(xy[~link, 0], xy[~link, 1], s=3, c="#ff7f0e", label="no radio link", zorder=3)
    seen_goals = set()
    for k, dcs in enumerate(tr["decisions"]):
        c = KIND_COLOUR.get(dcs["kind"], "k")
        if dcs["goal"] is not None:
            key = (dcs["kind"], round(dcs["goal"][0], 0), round(dcs["goal"][1], 0))
            if key in seen_goals:
                continue
            seen_goals.add(key)
            ax.plot(dcs["goal"][0], dcs["goal"][1], "o", ms=5, color=c, alpha=0.85,
                    mec="w" if dcs["source"] == "rules" else "m", zorder=4)
    # annotate the first decision of every change of kind (except routine exploring)
    prev = None
    n_note = 0
    for dcs in tr["decisions"]:
        if dcs["kind"] != prev and dcs["kind"] not in ("EXPLORE",) and n_note < 9:
            n_note += 1
            ax.annotate("%d" % n_note, (dcs["x"], dcs["y"]), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color=KIND_COLOUR.get(dcs["kind"], "k"), weight="bold",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="w", ec=KIND_COLOUR.get(dcs["kind"], "k")))
            dcs["_note"] = n_note
        prev = dcs["kind"]
    for kind, c in KIND_COLOUR.items():
        ax.plot([], [], "o", color=c, label="goal: " + kind)
    for p in w.persons:
        found = p["found_t"] is not None
        ax.plot(p["x"], p["y"], "*", ms=16, color="#2ca02c" if found else "#d62728", mec="k",
                label=None, zorder=5)
    ax.plot([], [], "*", ms=12, color="#2ca02c", mec="k", label="person found")
    ax.plot([], [], "*", ms=12, color="#d62728", mec="k", label="person not found")
    for s in w.sightings:
        if s["false"]:
            ax.plot(s["x"], s["y"], "x", ms=8, color="#9467bd")
    for (bx, by, br) in w.blocked:
        ax.add_patch(plt.Circle((bx, by), br, fill=False, ec="#8c564b", lw=2, ls="--"))
    ax.plot(w.home[0], w.home[1], "s", ms=12, color="#2ca02c", mec="k", label="home", zorder=6)
    ax.plot(w.x, w.y, "^", ms=12, color="#d62728", mec="k", label="end pose", zorder=6)
    if w.rockfall_at is not None:
        ax.annotate("rockfall t=%.0f s" % w.rockfall_at[2], (w.rockfall_at[0], w.rockfall_at[1]),
                    xytext=(10, 10), textcoords="offset points", color="#8c564b", fontsize=9)
    notes = []
    for d in tr["decisions"]:
        if "_note" in d:
            notes.append("%d  t=%5.0fs %-14s %s" % (d["_note"], d["t"], d["kind"], d["reason"]))
    ax.set_title("%s mine seed %d | %s, on_link_loss %s, on_person_found %s, first_branch %s\n"
                 "outcome %s | found %d/%d, reported %d | returned %s | lost %s | %.0f s, %.0f m | score %.1f"
                 % (a.style, a.seed, m["task"], m["on_link_loss"], m["on_person_found"], m["first_branch"],
                    m["outcome"], m["found"], m["persons"], m["reported"], m["returned"], m["lost"],
                    m["time_s"], m["distance_m"], m["score"]), fontsize=10)
    import textwrap
    wrapped = []
    for n in notes:
        wrapped.extend(textwrap.wrap(n, 150, subsequent_indent=" " * 30))
    axn.text(0.0, 1.0, "Numbered decisions (first of each change of kind; routine EXPLORE steps omitted):\n" +
             "\n".join(wrapped), transform=axn.transAxes, fontsize=7.5, family="monospace", va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, framealpha=0.85)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)
    keep = dict((k, v) for k, v in m.items() if k not in ("trace", "world"))
    print(json.dumps(keep, indent=1))
    print("decisions:")
    for d in tr["decisions"]:
        print("  t=%6.0f %-14s %-10s %-13s %s" % (d["t"], d["kind"], d["sensor"], d["tier"], d["reason"]))


if __name__ == "__main__":
    main()
