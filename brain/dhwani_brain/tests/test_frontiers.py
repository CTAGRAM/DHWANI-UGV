# -*- coding: utf-8 -*-
"""Frontier detection and clustering, goal snapping, search (camera coverage) goals."""
from __future__ import division, print_function, absolute_import

import math
import unittest

import numpy as np

from dhwani_brain import Brain
from dhwani_brain.mapview import MapView
from dhwani_brain.geodesic import Graph, distances
from dhwani_brain.frontiers import find_frontiers
from dhwani_brain.tests.synth import Mine, t_junction, obs_for


def frontiers_of(mine, reveal, robot):
    g = mine.grid(reveal)
    v = MapView(g, mine.res, (0.0, 0.0), 0.2, 0.15, include=[robot])
    rc = v.snap(robot[0], robot[1], 1.0, v.passable)
    gr = Graph(v.passable)
    d = distances(gr, [gr.node(rc)])[0]
    frs, _ = find_frontiers(v, gr.image(d), robot)
    return v, gr, d, frs


class TestFrontiers(unittest.TestCase):
    def test_open_corridor_end_is_one_frontier(self):
        m = Mine(30, 6, [(1, 2, 29, 4)])
        v, gr, d, frs = frontiers_of(m, [(0, 0, 15, 6)], (5.0, 3.0))
        self.assertEqual(len(frs), 1)
        f = frs[0]
        self.assertTrue(13.5 <= f.gx <= 15.2, f.gx)          # on the near lip of the opening
        self.assertTrue(v.passable[f.goal_cell])              # a place the robot centre may stand
        self.assertTrue(np.isfinite(d[gr.node(f.goal_cell)]))
        self.assertLess(abs(math.atan2(f.gy - 3.0, f.gx - 5.0)), 0.2)   # straight ahead (east)
        self.assertGreaterEqual(f.size_m, 1.2)

    def test_t_junction_two_frontiers_with_bearings(self):
        m = t_junction()
        v, gr, d, frs = frontiers_of(m, [(8.5, 0.5, 11.5, 24), (5, 20.5, 15, 23.5)], (10.0, 18.0))
        xs = sorted(f.gx for f in frs)
        self.assertEqual(len(frs), 2, xs)
        self.assertLess(xs[0], 6.5)
        self.assertGreater(xs[1], 13.5)

    def test_speckle_is_not_a_frontier(self):
        m = Mine(20, 6, [(1, 2, 19, 4)])
        g = m.grid(None)
        g[30, 100] = -1                                       # one unknown 0.1 m cell in the corridor
        v = MapView(g, 0.1, (0, 0), 0.2, 0.15, include=[(5.0, 3.0)])
        gr = Graph(v.passable)
        rc = v.snap(5.0, 3.0, 1.0, v.passable)
        dd = distances(gr, [gr.node(rc)])[0]
        frs, _ = find_frontiers(v, gr.image(dd), (5.0, 3.0))
        self.assertEqual(len(frs), 0)

    def test_goals_are_reachable_passable_and_not_in_unknown(self):
        m = t_junction()
        v, gr, d, frs = frontiers_of(m, [(8.5, 0.5, 11.5, 24), (5, 20.5, 15, 23.5)], (10.0, 18.0))
        for f in frs:
            self.assertTrue(v.passable[f.goal_cell])
            self.assertFalse(v.unknown[f.goal_cell])

    def test_search_goal_for_unseen_dead_end(self):
        # fully mapped T; the robot only drove up the stem, so the arm ends were never within 3 m
        m = t_junction()
        o = obs_for(m, (10.0, 18.0, math.pi / 2), home=(10.0, 2.0), reveal=None,
                    trail=[(10.0, y) for y in np.arange(2.0, 18.5, 0.5)])
        b = Brain({"task": "SEARCH_SURVIVORS"})
        d = b.decide(o)
        self.assertEqual(d["kind"], "EXPLORE")
        ctx = b.last["ctx"]
        self.assertEqual(len(ctx.frontiers), 0)                # lidar has mapped everything
        self.assertEqual(len(ctx.searches), 1)                 # arms + junction: one unseen region
        # after driving up to the junction the two arm ends are separate unseen regions
        o2 = obs_for(m, (10.0, 22.0, math.pi / 2), home=(10.0, 2.0), reveal=None,
                     trail=[(10.0, y) for y in np.arange(2.0, 22.5, 0.5)])
        b2 = Brain({"task": "SEARCH_SURVIVORS"})
        d = b2.decide(o2)
        xs = sorted(c.goal[0] for c in b2.last["ctx"].searches)
        self.assertEqual(len(xs), 2, xs)
        self.assertLess(xs[0], 6.0)
        self.assertGreater(xs[1], 14.0)
        self.assertEqual(d["kind"], "EXPLORE")
        # MAP_AREA does not do camera search: nothing left -> go home
        d2 = Brain({"task": "MAP_AREA"}).decide(o)
        self.assertEqual(d2["kind"], "RETURN_HOME")


if __name__ == "__main__":
    unittest.main()
