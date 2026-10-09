#!/usr/bin/env python
"""Benchmark the on-board decision brain ON the rover (Python 2.7, numpy 1.13), on the rover's
real SLAM map, the way the executive calls it.

It grabs the live /map and the robot pose once, then times Brain.decide() over many calls with
the rules policy and, if present, the learned policy. Use --load to add two busy processes that
emulate the CPU the rover uses while driving.

    python bench_brain.py [--load] [--n 300] [--out results.json]
"""
from __future__ import division, print_function

import argparse
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dhwani_brain"))


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


def grab_map_and_pose():
    import rospy
    import tf
    from nav_msgs.msg import OccupancyGrid
    rospy.init_node("bench_brain", anonymous=True, disable_signals=True)
    m = rospy.wait_for_message("/map", OccupancyGrid, timeout=15)
    tfl = tf.TransformListener()
    tfl.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(5.0))
    (t, q) = tfl.lookupTransform("map", "base_footprint", rospy.Time(0))
    yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
    g = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
    return g, m.info.resolution, (m.info.origin.position.x, m.info.origin.position.y), (t[0], t[1], yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from dhwani_brain import Brain
    grid, res, origin, pose = grab_map_and_pose()
    mission = {"task": "SEARCH_SURVIVORS", "understood": True, "report_person": True, "approach_person": True,
               "on_person_found": "REPORT_AND_CONTINUE", "on_link_loss": "CONTINUE_THEN_RETURN",
               "return_when_done": True, "first_branch": "ANY", "speed": "NORMAL",
               "max_duration_min": None, "max_distance_m": None, "min_battery_pct": None}
    obs = {"grid": grid, "res": res, "origin": origin, "pose": pose, "home": (pose[0], pose[1]),
           "battery_pct": 72.0, "link_ok": False, "since_contact_s": 40.0, "elapsed_s": 300.0, "odo_m": 35.0,
           "persons": [{"x": pose[0] + 2.0, "y": pose[1] + 1.0, "confirmed": False, "reported": False, "hits": 1}],
           "blocked": [], "last_link_pose": (pose[0], pose[1]), "trail": [(pose[0], pose[1])]}
    stop = multiprocessing.Event()
    burners = []
    if a.load:
        for _ in range(2):
            p = multiprocessing.Process(target=burn, args=(stop,))
            p.daemon = True
            p.start()
            burners.append(p)
        time.sleep(20)
    out = {"condition": "emulated driving load (+2 busy cores)" if a.load else "rover parked, full ROS stack",
           "map_cells": [int(grid.shape[0]), int(grid.shape[1])], "res": res, "loadavg_start": loadavg(), "policies": {}}
    for learned in (False, True):
        r0 = rss_mb()
        try:
            b = Brain(mission, {"use_learned": learned})
        except Exception as e:
            out["policies"]["learned" if learned else "rules"] = {"error": repr(e)}
            continue
        d = b.decide(obs)
        r1 = rss_mb()
        ts = []
        for i in range(a.n):
            t = time.time()
            d = b.decide(obs)
            ts.append((time.time() - t) * 1000.0)
        out["policies"]["learned" if learned else "rules"] = {
            "median_ms": round(pct(ts, 50), 2), "p95_ms": round(pct(ts, 95), 2), "p99_ms": round(pct(ts, 99), 2),
            "max_ms": round(max(ts), 2), "rss_added_mb": round(r1 - r0, 1), "n": a.n,
            "decision": {"kind": d.get("kind"), "source": d.get("source"), "reason": d.get("reason"),
                         "n_candidates": len(d.get("candidates") or [])}}
    out["loadavg_end"] = loadavg()
    stop.set()
    for p in burners:
        p.join(1)
    js = json.dumps(out, indent=1, default=str)
    print(js)
    if a.out:
        with open(a.out, "w") as f:
            f.write(js)


if __name__ == "__main__":
    main()
