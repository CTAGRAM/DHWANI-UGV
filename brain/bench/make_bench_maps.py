# -*- coding: utf-8 -*-
"""Build the synthetic 60 m x 60 m benchmark maps at the rover's gmapping resolution (0.05 m).

Run on the Mac:  python3 bench/make_bench_maps.py   -> bench/bench_maps.npz
The rover benchmark (bench_brain.py) only loads the result, so the Nano never runs the generator.

Each map is a generated mine, 60 m x 60 m, rendered as a ROS OccupancyGrid would be after an
exploration: everything west of x = 42 m is known (free 0, walls 100), the rest unknown, so the
brain sees real frontiers along the cut. The robot stands at the known cell farthest (geodesic)
from home, which makes the robot and home wavefronts as long as they get on this map.
"""
from __future__ import division, print_function, absolute_import

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dhwani_brain.sim.mapgen import generate           # noqa: E402
from dhwani_brain.geodesic import Graph, distances       # noqa: E402


def known_grid(free, res, cut_x):
    near = free.copy()
    for _ in range(2):
        n = near.copy()
        n[1:, :] |= near[:-1, :]
        n[:-1, :] |= near[1:, :]
        n[:, 1:] |= near[:, :-1]
        n[:, :-1] |= near[:, 1:]
        near = n
    g = np.full(free.shape, -1, dtype=np.int8)
    g[free] = 0
    g[near & ~free] = 100
    g[:, int(cut_x / res):] = -1
    return g


def main():
    out = {}
    for name, style, seed in (("drift60", "drift", 7), ("lattice60", "lattice", 11)):
        m = generate(style, seed, res=0.05, extent=(60.0, 60.0))
        g = known_grid(m.free, m.res, 42.0)
        # robot: known free cell farthest from home (geodesic, coarse 0.2 m)
        f = 4
        H, W = (g.shape[0] // f) * f, (g.shape[1] // f) * f
        cf = (g[:H, :W] == 0).reshape(H // f, f, W // f, f).all(axis=3).all(axis=1)
        gr = Graph(cf)
        hi, hj = int(m.home[1] / 0.2), int(m.home[0] / 0.2)
        hn = gr.node((hi, hj))
        if hn < 0:
            ii, jj = gr.rr, gr.cc
            hn = int(np.argmin((ii - hi) ** 2 + (jj - hj) ** 2))
        d = distances(gr, [hn])[0]
        d[~np.isfinite(d)] = -1
        k = int(np.argmax(d))
        rx, ry = (gr.cc[k] + 0.5) * 0.2, (gr.rr[k] + 0.5) * 0.2
        print("%s: grid %s, known free %.0f m2, robot (%.1f, %.1f) is %.0f m from home" %
              (name, g.shape, (g == 0).sum() * 0.0025, rx, ry, d[k] * 0.2))
        out[name + "_grid"] = g
        out[name + "_meta"] = np.array([0.05, 0.0, 0.0, rx, ry, 0.0, m.home[0], m.home[1]], dtype=np.float64)
    np.savez_compressed(os.path.join(HERE, "bench_maps.npz"), **out)
    print("wrote", os.path.join(HERE, "bench_maps.npz"))


if __name__ == "__main__":
    main()
