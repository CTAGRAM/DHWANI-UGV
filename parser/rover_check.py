#!/usr/bin/env python3
"""Rover compatibility check (Python 3.6, numpy 1.13). Not the formal benchmark.

Loads both models, parses 10 hand-written test sentences (chosen before any test result was
seen), prints the JSON, then times 50 calls of the example sentence after 5 warm-up calls.

    cd ~/dhwani_parser && python3 rover_check.py > rover_check_output.json
"""
from __future__ import division, print_function

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dhwani_parser import MissionParser  # noqa: E402

SENTENCES = [
    "Go through this mine and find any person alive, report where they are, and come back if "
    "the link becomes unstable",
    "stop",
    "reverse back",
    "go to point B",
    "survey the mine for 45 min, keep 30% battery reserve",
    "serch for survivers in the mine",
    "order pizza",
    "search for one and a half hours",
    "Look for survivors in the left branch first, go slowly, and don't go beyond 200 metres",
    "Rover, there has been a roof fall in panel 4. Three miners are missing. Search the area, "
    "move slowly, and report their positions. If communication is lost, return immediately.",
]
TIMED = SENTENCES[0]


def main():
    out = {"host": os.uname()[1], "python": sys.version.split()[0], "numpy": np.__version__,
           "loadavg_before": [round(x, 2) for x in os.getloadavg()], "models": {}}
    for name in ("tinybert4", "berttiny"):
        t0 = time.time()
        p = MissionParser(os.path.join(HERE, "models", name))
        load_s = time.time() - t0
        parsed = [p.parse(s) for s in SENTENCES]
        for _ in range(5):
            p.parse(TIMED)
        ts = []
        for _ in range(50):
            t = time.time()
            p.parse(TIMED)
            ts.append((time.time() - t) * 1e3)
        short = []
        for _ in range(20):
            t = time.time()
            p.parse("stop")
            short.append((time.time() - t) * 1e3)
        n_ids = len(p.tok.encode(TIMED)[0])
        out["models"][name] = {
            "load_s": round(load_s, 2),
            "timed_sentence_ids": n_ids,
            "padded_len": min(64, -(-n_ids // 8) * 8),
            "median_ms_50_calls": round(float(np.median(ts)), 1),
            "min_ms": round(float(np.min(ts)), 1), "max_ms": round(float(np.max(ts)), 1),
            "median_ms_stop_20_calls": round(float(np.median(short)), 1),
            "parsed": parsed}
    out["loadavg_after"] = [round(x, 2) for x in os.getloadavg()]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
