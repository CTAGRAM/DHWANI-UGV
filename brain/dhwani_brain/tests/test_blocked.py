# -*- coding: utf-8 -*-
"""A corridor closed behind the rover must yield another route or a HOLD, never a goal or a path
through the rubble."""
from __future__ import division, print_function, absolute_import

import math
import unittest

from dhwani_brain import Brain
from dhwani_brain.tests.synth import loop_mine, dead_end_line, Mine, obs_for

RUBBLE = (15.0, 3.0, 1.0)             # closes the south corridor of the loop / the straight line


def clear_of(pt, blk, margin=0.0):
    return math.hypot(pt[0] - blk[0], pt[1] - blk[1]) > blk[2] + margin


class TestBlocked(unittest.TestCase):
    def check_no_rubble(self, d, blk):
        if d["goal"] is not None:
            self.assertTrue(clear_of(d["goal"], blk, 0.15), "goal inside the blocked region: %r" % (d["goal"],))
        for p in d["path"]:
            self.assertTrue(clear_of(p, blk, 0.1), "path crosses the blocked region at %r" % (p,))
        for c in d["candidates"]:
            if c["goal"] is not None:
                self.assertTrue(clear_of(c["goal"], blk, 0.15), "candidate inside rubble: %r" % (c,))

    def test_loop_goes_around(self):
        m = loop_mine()
        o = obs_for(m, (27.0, 12.0, math.pi / 2), home=(3.0, 3.0), blocked=[RUBBLE], battery_pct=12.0)
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        self.assertTrue(d["status"]["home_reachable"])
        self.assertGreater(d["status"]["d_home_m"], 55.0)           # the long way round (~57 m)
        self.check_no_rubble(d, RUBBLE)
        self.assertTrue(any(p[1] > 25.0 for p in d["path"]))       # through the north corridor

    def test_without_rubble_takes_the_short_way(self):
        m = loop_mine()
        o = obs_for(m, (27.0, 12.0, math.pi / 2), home=(3.0, 3.0), battery_pct=12.0)
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertEqual(d["kind"], "RETURN_HOME")
        self.assertLess(d["status"]["d_home_m"], 40.0)

    def test_dead_end_holds(self):
        m = dead_end_line()
        o = obs_for(m, (30.0, 3.0, 0.0), home=(2.0, 3.0), blocked=[RUBBLE], battery_pct=60.0)
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertEqual(d["kind"], "HOLD", d["reason"])
        self.assertEqual(d["tier"], "safety")
        self.assertIn("no known route home", d["reason"])
        self.check_no_rubble(d, RUBBLE)

    def test_explores_for_a_route_within_reserve_only(self):
        # line continues east into unknown: a possible way round, cheap enough -> explore it
        m = Mine(60, 6, [(1, 2, 59, 4)])
        rev = [(0, 0, 36, 6)]
        o = obs_for(m, (30.0, 3.0, 0.0), home=(2.0, 3.0), reveal=rev, blocked=[RUBBLE], battery_pct=60.0)
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertEqual(d["kind"], "EXPLORE", d["reason"])
        self.assertEqual(d["tier"], "safety")
        self.assertGreater(d["goal"][0], 33.0)
        self.check_no_rubble(d, RUBBLE)
        # same, but the battery cannot afford the detour -> hold instead of gambling
        o["battery_pct"] = 9.0
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertEqual(d["kind"], "HOLD", d["reason"])
        self.check_no_rubble(d, RUBBLE)

    def test_route_search_ignores_mission_distance_budget(self):
        m = Mine(60, 6, [(1, 2, 59, 4)])
        o = obs_for(m, (30.0, 3.0, 0.0), home=(2.0, 3.0), reveal=[(0, 0, 36, 6)], blocked=[RUBBLE],
                    battery_pct=60.0, odo_m=95.0)
        d = Brain({"task": "SEARCH_SURVIVORS", "max_distance_m": 100}).decide(o)
        self.assertEqual(d["kind"], "EXPLORE", d["reason"])

    def test_rubble_over_home_is_unreachable(self):
        m = dead_end_line()
        o = obs_for(m, (30.0, 3.0, 0.0), home=(2.0, 3.0), blocked=[(2.0, 3.0, 1.2)], battery_pct=60.0)
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertIn(d["kind"], ("HOLD", "EXPLORE"))
        self.assertNotEqual(d["kind"], "RETURN_HOME")


if __name__ == "__main__":
    unittest.main()
