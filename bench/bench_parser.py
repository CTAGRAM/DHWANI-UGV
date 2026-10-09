#!/usr/bin/env python3
"""Benchmark the plain-English mission parser ON the rover (Python 3.6, numpy 1.13, CPU only).

Measures, for each model in ~/dhwani_parser/models: load time, resident memory added by loading,
and the end-to-end parse time (tokenise + encoder + heads + JSON) over real instructions, as
median / p95 / p99 / max. Run it twice: with the rover parked, and with --load, which adds two
busy processes to emulate the CPU the rover uses while driving (load average 6.6 to 9.4 measured).

    python3 bench_parser.py [--load] [--n 200] [--out results.json]
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/dhwani_parser"))

SENTENCES = [
    "Go through this mine and find any person alive, report where they are, and come back if the link becomes unstable",
    "stop",
    "come back home",
    "search the left tunnel for survivors for 20 minutes",
    "map this area and return when battery drops below 30 percent",
    "go to point B",
    "kindly check if anybody is trapped near the collapsed section and report",
    "explore everything you can reach but dont go further than 100 metres",
    "find the missing miner, stay with him if you find him",
    "halt now",
    "what is the weather today",
    "go slowly and look for people, if you lose signal come back immediately",
]


def rss_mb():
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024.0
    return 0.0


def burn(stop):
    while not stop.is_set():
        pass


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1))))]


def loadavg():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from dhwani_parser import MissionParser
    stop = multiprocessing.Event()
    burners = []
    if a.load:
        for _ in range(2):
            p = multiprocessing.Process(target=burn, args=(stop,))
            p.daemon = True
            p.start()
            burners.append(p)
        time.sleep(20)          # let the load average climb before measuring
    out = {"condition": "emulated driving load (+2 busy cores)" if a.load else "rover parked, full ROS stack",
           "loadavg_start": loadavg(), "models": {}}
    root = os.path.expanduser("~/dhwani_parser/models")
    for name in sorted(os.listdir(root)):
        mdir = os.path.join(root, name)
        if not os.path.isdir(mdir):
            continue
        r0 = rss_mb()
        t0 = time.time()
        p = MissionParser(mdir)
        load_s = time.time() - t0
        for s in SENTENCES[:4]:
            p.parse(s)                      # warm-up
        r1 = rss_mb()
        ts = []
        for i in range(a.n):
            s = SENTENCES[i % len(SENTENCES)]
            t = time.time()
            p.parse(s)
            ts.append((time.time() - t) * 1000.0)
        out["models"][name] = {"load_s": round(load_s, 2), "rss_added_mb": round(r1 - r0, 1),
                               "median_ms": round(pct(ts, 50), 1), "p95_ms": round(pct(ts, 95), 1),
                               "p99_ms": round(pct(ts, 99), 1), "max_ms": round(max(ts), 1), "n": a.n,
                               "example": p.parse(SENTENCES[0])}
        del p
    out["loadavg_end"] = loadavg()
    stop.set()
    for b in burners:
        b.join(1)
    js = json.dumps(out, indent=1, default=str)
    print(js)
    if a.out:
        with open(a.out, "w") as f:
            f.write(js)


if __name__ == "__main__":
    main()
