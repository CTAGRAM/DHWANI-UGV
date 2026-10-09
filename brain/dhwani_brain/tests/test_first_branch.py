# -*- coding: utf-8 -*-
"""first_branch: LEFT / RIGHT / STRAIGHT relative to the heading when the mission started."""
from __future__ import division, print_function, absolute_import

import math
import unittest

from dhwani_brain import Brain
from dhwani_brain.tests.synth import t_junction, cross_junction, obs_for

T_REVEAL = [(8.5, 0.5, 11.5, 24), (5, 20.5, 15, 23.5)]
X_REVEAL = [(10.5, 0.5, 13.5, 25), (7, 18.5, 17, 21.5)]


def decide(fb, mine, reveal, pose, start_yaw=math.pi / 2):
    b = Brain({"task": "SEARCH_SURVIVORS", "first_branch": fb}, {"start_yaw": start_yaw})
    return b, b.decide(obs_for(mine, pose, home=(pose[0], 2.0), reveal=reveal))


class TestFirstBranch(unittest.TestCase):
    def test_left_right_at_t(self):
        m = t_junction()
        _, d = decide("LEFT", m, T_REVEAL, (10.0, 18.0, math.pi / 2))
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertLess(d["goal"][0], 8.0, d["reason"])
        self.assertEqual(d["tier"], "mission")
        _, d = decide("RIGHT", m, T_REVEAL, (10.0, 18.0, math.pi / 2))
        self.assertGreater(d["goal"][0], 12.0, d["reason"])

    def test_straight_at_cross(self):
        m = cross_junction()
        _, d = decide("STRAIGHT", m, X_REVEAL, (12.0, 15.0, math.pi / 2))
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertGreater(d["goal"][1], 22.0, d["reason"])
        self.assertLess(abs(d["goal"][0] - 12.0), 1.5)

    def test_relative_to_start_heading_not_current(self):
        # started facing north; now momentarily facing east: LEFT still means west
        m = t_junction()
        _, d = decide("LEFT", m, T_REVEAL, (10.0, 18.0, 0.0))
        self.assertLess(d["goal"][0], 8.0, d["reason"])

    def test_commits_once(self):
        m = t_junction()
        b, d = decide("LEFT", m, T_REVEAL, (10.0, 18.0, math.pi / 2))
        self.assertTrue(b.state.get("branch_committed"))
        d = b.decide(obs_for(m, (10.0, 18.0, math.pi / 2), home=(10.0, 2.0), reveal=T_REVEAL))
        self.assertEqual(d["tier"], "discretionary")

    def test_any_is_not_forced(self):
        m = t_junction()
        _, d = decide("ANY", m, T_REVEAL, (10.0, 18.0, math.pi / 2))
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertEqual(d["tier"], "discretionary")


if __name__ == "__main__":
    unittest.main()
