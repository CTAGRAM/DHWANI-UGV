# -*- coding: utf-8 -*-
"""Run the brain on a REAL map the rover built (map_server PGM + YAML), not a simulated one.

    python bench/real_map_check.py [../maps/tour_20260923.yaml]

Converts the PGM the way map_server would publish it (row 0 = minimum y, so the image is
flipped; 254+ free, <= 50 occupied, the rest unknown), stands the robot on known free floor
and prints the decision, the candidates and the timing for a SEARCH_SURVIVORS mission.
"""
from __future__ import division, print_function, absolute_import

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dhwani_brain import Brain                         # noqa: E402


def load_map(yaml_path):
    meta = {}
    for line in open(yaml_path):
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    res = float(meta["resolution"])
    origin = [float(x) for x in meta["origin"].strip("[]").split(",")[:2]]
    pgm = os.path.join(os.path.dirname(yaml_path), meta["image"])
    data = open(pgm, "rb").read()
    # P5 header: magic, optional comments, width height, maxval
    parts, pos = [], 0
    while len(parts) < 4:
        end = data.index(b"\n", pos)
        line = data[pos:end].strip()
        pos = end + 1
        if line.startswith(b"#"):
            continue
        parts.extend(line.split())
    w, h = int(parts[1]), int(parts[2])
    img = np.frombuffer(data[pos:pos + w * h], dtype=np.uint8).reshape(h, w)
    grid = np.full((h, w), -1, dtype=np.int8)
    grid[img >= 250] = 0
    grid[img <= 50] = 100
    return grid[::-1].copy(), res, origin


def main():
    yp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(HERE)), "maps",
                                                            "tour_20260923.yaml")
    if not os.path.exists(yp):
        print("map not found: %s (skipping)" % yp)
        return
    grid, res, origin = load_map(yp)
    ys, xs = np.nonzero(grid == 0)
    k = len(xs) // 2
    x, y = origin[0] + (xs[k] + 0.5) * res, origin[1] + (ys[k] + 0.5) * res
    obs = {"grid": grid, "res": res, "origin": origin, "pose": (x, y, 0.0), "home": (x, y),
           "battery_pct": 80.0, "link_ok": True, "since_contact_s": 0.0, "elapsed_s": 60.0, "odo_m": 5.0,
           "persons": [], "blocked": [], "last_link_pose": (x, y), "trail": [(x, y)]}
    b = Brain({"task": "SEARCH_SURVIVORS"})
    for _ in range(3):
        d = b.decide(obs)
    print("map %s: %dx%d cells at %.3f m, %d known free" % (os.path.basename(yp), grid.shape[1], grid.shape[0],
                                                             res, int((grid == 0).sum())))
    print("robot at (%.2f, %.2f)" % (x, y))
    print("decision: %s %s %s" % (d["kind"], d["goal"], d["sensor"]))
    print("reason:   %s" % d["reason"])
    print("timing:   %s" % json.dumps(d["timing_ms"]))
    print("status:   %s" % json.dumps(d["status"]))
    for c in d["candidates"][:8]:
        print("  cand %-14s goal %-26s feasible %-5s d_robot %s" % (c["kind"], c["goal"], c["feasible"],
                                                                 c["d_robot_m"]))


if __name__ == "__main__":
    main()
