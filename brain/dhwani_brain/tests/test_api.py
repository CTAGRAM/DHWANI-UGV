# -*- coding: utf-8 -*-
"""The Brain contract: decision keys and types, JSON-safety, task handling, persons, robustness,
learned-policy gating."""
from __future__ import division, print_function, absolute_import

import json
import math
import os
import unittest

import numpy as np

from dhwani_brain import Brain
from dhwani_brain.features import NC
from dhwani_brain.learned import default_weights_path
from dhwani_brain.tests.synth import Mine, t_junction, obs_for

KINDS = ("EXPLORE", "VERIFY", "RETURN_HOME", "RETURN_TO_LINK", "HOLD", "DONE")


def corridor(**kw):
    m = Mine(60, 6, [(1, 2, 59, 4)])
    o = obs_for(m, (20.0, 3.0, 0.0), home=(2.0, 3.0), reveal=[(0, 0, 25, 6)])
    o.update(kw)
    return o


class TestContract(unittest.TestCase):
    def check(self, d):
        for k in ("kind", "goal", "sensor", "reason", "source", "confidence", "candidates"):
            self.assertIn(k, d)
        self.assertIn(d["kind"], KINDS)
        self.assertIn(d["sensor"], ("MOVE", "DWELL_SCAN"))
        self.assertIn(d["source"], ("rules", "learned"))
        self.assertTrue(isinstance(d["reason"], str) and len(d["reason"]) > 5)
        if d["goal"] is not None:
            self.assertEqual(len(d["goal"]), 3)
            self.assertTrue(all(isinstance(v, float) for v in d["goal"]))
        for c in d["candidates"]:
            for k in ("kind", "goal", "score", "features"):
                self.assertIn(k, c)
            self.assertEqual(len(c["features"]), NC)
        json.dumps(d)                                   # publishable on /executive/state as-is

    def test_basic_decision(self):
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(corridor())
        self.check(d)
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertEqual(d["source"], "rules")
        self.assertGreater(len(d["path"]), 3)

    def test_rover_resolution_grid(self):
        m = Mine(40, 6, [(1, 2, 39, 4)], res=0.05)
        o = obs_for(m, (10.0, 3.0, 0.0), home=(2.0, 3.0), reveal=[(0, 0, 25, 6)])
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.check(d)
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertGreater(d["goal"][0], 22.0)

    def test_origin_offset(self):
        o = corridor()
        o["origin"] = (-25.0, 7.0)
        o["pose"] = (20.0 - 25.0, 3.0 + 7.0, 0.0)
        o["home"] = (2.0 - 25.0, 3.0 + 7.0)
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(o)
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertAlmostEqual(d["goal"][1], 10.0, delta=1.0)

    def test_tasks(self):
        self.assertEqual(Brain({"task": "HALT"}).decide(corridor())["kind"], "HOLD")
        d = Brain({"task": "UNSUPPORTED", "understood": False}).decide(corridor())
        self.assertEqual(d["kind"], "HOLD")
        self.assertIn("not understood", d["reason"])
        self.assertEqual(Brain({"task": "RETURN_HOME"}).decide(corridor())["kind"], "RETURN_HOME")
        self.assertEqual(Brain({"task": "RETURN_HOME"}).decide(corridor(pose=(2.0, 3.0, 0.0)))["kind"], "DONE")
        self.assertEqual(Brain({}).decide(corridor())["kind"], "HOLD")

    def test_goto_waypoint(self):
        wps = {"A": {"x": 15.0, "y": 3.0, "yaw_deg": 0.0}}
        d = Brain({"task": "GOTO_WAYPOINT", "waypoint": "A"}, {"waypoints": wps}).decide(corridor())
        self.assertEqual(d["kind"], "EXPLORE")
        self.assertEqual(d.get("target"), "WAYPOINT")
        self.assertLess(abs(d["goal"][0] - 15.0), 0.5)
        d = Brain({"task": "GOTO_WAYPOINT", "waypoint": "B"}, {"waypoints": wps}).decide(corridor())
        self.assertEqual(d["kind"], "HOLD")
        d = Brain({"task": "GOTO_WAYPOINT", "waypoint": "A"}, {"waypoints": wps}).decide(
            corridor(pose=(15.0, 3.0, 0.0)))
        self.assertEqual(d["kind"], "DONE")

    def test_bad_obs_holds_instead_of_raising(self):
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide({"pose": (0, 0, 0)})
        self.assertEqual(d["kind"], "HOLD")
        self.assertIn("brain error", d["reason"])

    def test_unknown_param_rejected(self):
        self.assertRaises(KeyError, Brain, {"task": "SEARCH_SURVIVORS"}, {"no_such_param": 1})

    def test_unicode_mission_fields(self):
        m = json.loads(u'{"task": "SEARCH_SURVIVORS", "first_branch": "left", "on_link_loss": "hold"}')
        b = Brain(m)
        self.assertEqual(b.mission["first_branch"], "LEFT")
        self.assertEqual(b.mission["on_link_loss"], "HOLD")


class TestPersons(unittest.TestCase):
    def test_verify_unconfirmed_sighting(self):
        p = [{"x": 12.0, "y": 3.0, "confirmed": False, "reported": False, "hits": 1}]
        b = Brain({"task": "SEARCH_SURVIVORS"})
        d = b.decide(corridor(persons=p))
        self.assertEqual(d["kind"], "VERIFY", d["reason"])
        self.assertEqual(d["sensor"], "DWELL_SCAN")
        g = d["goal"]
        self.assertLess(math.hypot(g[0] - 12.0, g[1] - 3.0), 2.6)
        # standing at the standoff, still unconfirmed: look again, then give up
        kinds = []
        for _ in range(3):
            d = b.decide(corridor(persons=p, pose=(g[0], g[1], g[2])))
            kinds.append(d["kind"])
        self.assertEqual(kinds[0], "VERIFY")
        self.assertNotEqual(kinds[-1], "VERIFY")

    def test_map_area_does_not_approach(self):
        p = [{"x": 12.0, "y": 3.0, "confirmed": False, "reported": False, "hits": 1}]
        d = Brain({"task": "MAP_AREA"}).decide(corridor(persons=p))
        self.assertEqual(d["kind"], "EXPLORE")

    def test_report_and_return(self):
        p = [{"x": 12.0, "y": 3.0, "confirmed": True, "reported": True, "hits": 5}]
        b = Brain({"task": "SEARCH_SURVIVORS", "on_person_found": "REPORT_AND_RETURN"})
        self.assertEqual(b.decide(corridor(persons=p))["kind"], "RETURN_HOME")
        self.assertEqual(b.decide(corridor())["kind"], "RETURN_HOME")          # latched

    def test_stay(self):
        p = [{"x": 12.0, "y": 3.0, "confirmed": True, "reported": True, "hits": 5}]
        d = Brain({"task": "SEARCH_SURVIVORS", "on_person_found": "STAY"}).decide(corridor(persons=p))
        self.assertEqual(d["kind"], "HOLD", d["reason"])
        self.assertIn("stay", d["reason"])
        self.assertLess(math.hypot(d["goal"][0] - 12.0, d["goal"][1] - 3.0), 2.6)

    def test_report_and_continue_keeps_exploring_when_reported(self):
        p = [{"x": 12.0, "y": 3.0, "confirmed": True, "reported": True, "hits": 5}]
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(corridor(persons=p))
        self.assertEqual(d["kind"], "EXPLORE")


class TestLearnedGate(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(default_weights_path()):
            self.skipTest("no packaged weights")

    def test_threshold_above_one_means_rules(self):
        b = Brain({"task": "SEARCH_SURVIVORS"}, {"use_learned": True, "learned_threshold": 1.01})
        self.assertIsNone(b.policy_error)
        d = b.decide(corridor())
        self.assertEqual(d["source"], "rules")
        self.assertIn("learned", d)                        # it ran, but was not trusted

    def test_learned_only_on_discretionary(self):
        b = Brain({"task": "SEARCH_SURVIVORS"}, {"use_learned": True, "learned_threshold": 0.0})
        d = b.decide(corridor())
        self.assertIn(d["source"], ("rules", "learned"))
        d = b.decide(corridor(battery_pct=3.0))
        self.assertEqual(d["source"], "rules")           # safety: always rules
        self.assertEqual(d["kind"], "RETURN_HOME")

    def test_missing_weights_fall_back(self):
        b = Brain({"task": "SEARCH_SURVIVORS"}, {"use_learned": True, "weights": "/nonexistent.npz"})
        self.assertIsNotNone(b.policy_error)
        d = b.decide(corridor())
        self.assertEqual(d["source"], "rules")
        self.assertEqual(d["kind"], "EXPLORE")


if __name__ == "__main__":
    unittest.main()
