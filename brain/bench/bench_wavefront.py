# -*- coding: utf-8 -*-
"""Wavefront benchmark on the synthetic 60 m x 60 m maps (bench_maps.npz). Short: ~30-60 s.

    python bench/bench_wavefront.py            (on the rover: python2.7)

Times, per map: pooling the 0.05 m grid to the 0.2 m planning grid, building the neighbour
table, the robot+home(+link) wavefront, and - for reference - the literal dense
dilation version of the same wavefront (one field). Reports medians over repeats.
"""
from __future__ import division, print_function, absolute_import

import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dhwani_brain.compat import clock                                      # noqa: E402
from dhwani_brain.mapview import MapView                                    # noqa: E402
from dhwani_brain.geodesic import Graph, distances, distances_dense         # noqa: E402


def med(f, n):
    ts = []
    out = None
    for _ in range(n):
        t = clock()
        out = f()
        ts.append((clock() - t) * 1e3)
    return float(np.median(ts)), out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    z = np.load(os.path.join(HERE, "bench_maps.npz"))
    res_all = {"python": platform.python_version(), "numpy": np.__version__,
               "machine": platform.machine(), "node": platform.node(), "maps": {}}
    for name in ("drift60", "lattice60"):
        g = z[name + "_grid"]
        meta = z[name + "_meta"]
        res, rx, ry, hx, hy = float(meta[0]), float(meta[3]), float(meta[4]), float(meta[6]), float(meta[7])
        t_map, view = med(lambda: MapView(g, res, (0.0, 0.0), 0.2, 0.15, include=[(rx, ry), (hx, hy)]), reps)
        rc = view.snap(rx, ry, 1.0, view.passable)
        hc = view.snap(hx, hy, 1.5, view.passable)
        t_graph, graph = med(lambda: Graph(view.passable), reps)
        rn, hn = graph.node(rc), graph.node(hc)
        t_two, D = med(lambda: distances(graph, [rn, hn]), reps)
        t_three, _ = med(lambda: distances(graph, [rn, hn, rn]), reps)
        t_one, _ = med(lambda: distances(graph, [rn]), reps)
        t_dense, Dd = med(lambda: distances_dense(view.passable, [rc]), 1)
        img = graph.image(D[0])
        same = bool(np.allclose(np.where(np.isfinite(img), img, -1), np.where(np.isfinite(Dd[0]), Dd[0], -1),
                                atol=1e-3))
        r = {"fine_grid": list(g.shape), "planning_grid": list(view.shape), "nodes": graph.N,
             "max_dist_m": round(float(np.max(D[0][np.isfinite(D[0])])) * view.cell, 1),
             "robot_home_m": round(float(D[0][hn]) * view.cell, 1),
             "pool_inflate_ms": round(t_map, 1), "neighbour_table_ms": round(t_graph, 1),
             "wavefront_1_field_ms": round(t_one, 1),
             "wavefront_robot_home_ms": round(t_two, 1),
             "wavefront_robot_home_link_ms": round(t_three, 1),
             "total_geodesic_ms (pool+table+2 fields)": round(t_map + t_graph + t_two, 1),
             "dense_dilation_1_field_ms": round(t_dense, 1), "dense_equals_sparse": same}
        res_all["maps"][name] = r
        print(name, json.dumps(r, sort_keys=True))
    print(json.dumps(res_all))


if __name__ == "__main__":
    main()
