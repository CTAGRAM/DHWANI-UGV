# -*- coding: utf-8 -*-
"""Geodesic wavefront: exact against a textbook Dijkstra, multi-source, cut-off, paths."""
from __future__ import division, print_function, absolute_import

import heapq
import unittest

import numpy as np

from dhwani_brain.geodesic import Graph, distances, distances_dense, descend, OFFSETS, SQ2


def dijkstra(P, src):
    H, W = P.shape
    d = np.full((H, W), np.inf)
    d[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        dd, (r, c) = heapq.heappop(pq)
        if dd > d[r, c]:
            continue
        for dr, dc in OFFSETS:
            a, b = r + dr, c + dc
            if not (0 <= a < H and 0 <= b < W) or not P[a, b]:
                continue
            if dr and dc and not (P[r + dr, c] and P[r, c + dc]):
                continue
            nd = dd + (SQ2 if dr and dc else 1.0)
            if nd < d[a, b] - 1e-9:
                d[a, b] = nd
                heapq.heappush(pq, (nd, (a, b)))
    return d


class TestWavefront(unittest.TestCase):
    def test_exact_vs_dijkstra_random_grids(self):
        rng = np.random.RandomState(1)
        for _ in range(25):
            P = rng.rand(rng.randint(5, 30), rng.randint(5, 30)) > rng.uniform(0.1, 0.45)
            g = Graph(P)
            if g.N < 2:
                continue
            s1, s2 = rng.randint(g.N), rng.randint(g.N)
            D = distances(g, [s1, s2, -1])
            for k, s in enumerate((s1, s2)):
                ref = dijkstra(P, (g.rr[s], g.cc[s]))
                img = g.image(D[k])
                self.assertTrue((np.isfinite(ref) == np.isfinite(img)).all())
                fin = np.isfinite(ref)
                self.assertLess(float(np.abs(ref[fin] - img[fin]).max()), 1e-3)
            self.assertTrue(np.isinf(D[2]).all())

    def test_dense_dilation_matches(self):
        rng = np.random.RandomState(2)
        P = rng.rand(25, 31) > 0.3
        g = Graph(P)
        s = int(g.N // 2)
        a = g.image(distances(g, [s])[0])
        b = distances_dense(P, [(g.rr[s], g.cc[s])])[0]
        self.assertTrue((np.isfinite(a) == np.isfinite(b)).all())
        fin = np.isfinite(a)
        self.assertLess(float(np.abs(a[fin] - b[fin]).max()), 1e-3)

    def test_multi_source_field_is_min_of_singles(self):
        rng = np.random.RandomState(3)
        P = rng.rand(20, 20) > 0.25
        g = Graph(P)
        srcs = [0, g.N // 3, g.N - 1]
        multi = distances(g, [srcs])[0]
        singles = distances(g, srcs)
        self.assertLess(float(np.nanmax(np.abs(np.where(np.isfinite(multi), multi, 0) -
                                                 np.where(np.isfinite(singles.min(0)), singles.min(0), 0)))), 1e-4)

    def test_max_dist_cutoff(self):
        P = np.ones((1, 50), dtype=bool)
        g = Graph(P)
        d = distances(g, [0], max_dist=10)[0]
        self.assertTrue(np.isfinite(d[:10]).all())
        self.assertTrue(np.isinf(d[12:]).all())

    def test_no_corner_cutting(self):
        P = np.array([[1, 0], [0, 1]], dtype=bool)          # diagonal pair touching at a corner
        g = Graph(P)
        d = distances(g, [0])[0]
        self.assertTrue(np.isinf(d[1]))

    def test_descend_is_a_shortest_path(self):
        rng = np.random.RandomState(4)
        P = rng.rand(30, 30) > 0.3
        g = Graph(P)
        d = distances(g, [0])[0]
        reach = np.flatnonzero(np.isfinite(d))
        for t in reach[::max(1, len(reach) // 15)]:
            path = descend(g, d, int(t))
            self.assertEqual(path[-1], 0)
            L = 0.0
            for a, b in zip(path[:-1], path[1:]):
                dr, dc = abs(int(g.rr[a]) - int(g.rr[b])), abs(int(g.cc[a]) - int(g.cc[b]))
                self.assertEqual(max(dr, dc), 1)
                L += SQ2 if dr and dc else 1.0
            self.assertAlmostEqual(L, float(d[t]), places=3)


if __name__ == "__main__":
    unittest.main()
