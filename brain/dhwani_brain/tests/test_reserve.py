# -*- coding: utf-8 -*-
"""Battery reserve, min_battery_pct, time and distance limits."""
from __future__ import division, print_function, absolute_import

import unittest

from dhwani_brain import Brain
from dhwani_brain.tests.synth import Mine, obs_for


def scene(**kw):
    """60 m corridor known to x = 36; robot at x = 30 (28 m from home), frontier 6 m ahead."""
    m = Mine(60, 6, [(1, 2, 59, 4)])
    o = obs_for(m, (30.0, 3.0, 0.0), home=(2.0, 3.0), reveal=[(0, 0, 36, 6)], battery_pct=80.0)
    o.update(kw)
    return o


def threshold(b, o):
    """The battery level at which this brain must turn back from this pose (from its own model)."""
    b2 = Brain(b.mission, dict((k, v) for k, v in b.params.items() if k in ("drain_pct_per_m",)))
    d = b2.decide(o)
    st = d["status"]
    la = b2.params["speed_mps"] * b2.params["redecide_s"]
    rate = st["rate_pct_per_m"]
    pf = b2.params["path_factor"]
    res = ((st["d_home_m"] + la) * pf * rate) * (1 + b2.params["reserve_frac"]) + b2.params["reserve_pct"]
    return res + la * pf * rate + st["floor_pct"]


class TestReserve(unittest.TestCase):
    def test_plenty_of_battery_explores(self):
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(scene())
        self.assertEqual(d["kind"], "EXPLORE", d["reason"])

    def test_below_reserve_returns(self):
        b = Brain({"task": "SEARCH_SURVIVORS"})
        th = threshold(b, scene())
        d = b.decide(scene(battery_pct=th - 0.5))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        self.assertEqual(d["tier"], "safety")
        self.assertIn("battery", d["reason"])

    def test_just_above_reserve_skips_frontier_that_would_break_it(self):
        b = Brain({"task": "SEARCH_SURVIVORS"})
        th = threshold(b, scene())
        d = b.decide(scene(battery_pct=th + 0.3))
        # not forced home yet, but the frontier 6 m deeper does not leave enough to come back
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        self.assertEqual(d["tier"], "discretionary")
        self.assertIn("battery", d["reason"])

    def test_min_battery_pct_is_a_floor(self):
        d = Brain({"task": "SEARCH_SURVIVORS", "min_battery_pct": 50}).decide(scene(battery_pct=55.0))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        d = Brain({"task": "SEARCH_SURVIVORS", "min_battery_pct": 20}).decide(scene(battery_pct=55.0))
        self.assertEqual(d["kind"], "EXPLORE", d["reason"])

    def test_measured_drain_overrides_optimistic_model(self):
        b = Brain({"task": "SEARCH_SURVIVORS"})
        b.decide(scene(battery_pct=90.0, odo_m=0.0))
        # 60 m driven cost 30 %: 0.5 %/m, six times the model; 28 m home now needs ~20 %
        d = b.decide(scene(battery_pct=60.0, odo_m=60.0))
        self.assertGreater(d["status"]["rate_pct_per_m"], 0.4)
        d = b.decide(scene(battery_pct=25.0, odo_m=130.0))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])

    def test_time_limit(self):
        d = Brain({"task": "SEARCH_SURVIVORS", "max_duration_min": 10}).decide(scene(elapsed_s=480.0))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        self.assertIn("time limit", d["reason"])
        d = Brain({"task": "SEARCH_SURVIVORS", "max_duration_min": 60}).decide(scene(elapsed_s=480.0))
        self.assertEqual(d["kind"], "EXPLORE")

    def test_distance_limit(self):
        d = Brain({"task": "SEARCH_SURVIVORS", "max_distance_m": 100}).decide(scene(odo_m=60.0))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        self.assertIn("distance limit", d["reason"])

    def test_limits_without_return_stop_in_place(self):
        d = Brain({"task": "SEARCH_SURVIVORS", "max_duration_min": 10, "return_when_done": False}).decide(
            scene(elapsed_s=601.0))
        self.assertEqual(d["kind"], "DONE", d["reason"])

    def test_at_home_low_battery_is_done(self):
        d = Brain({"task": "SEARCH_SURVIVORS"}).decide(scene(pose=(2.0, 3.0, 0.0), battery_pct=6.0))
        self.assertEqual(d["kind"], "DONE", d["reason"])


if __name__ == "__main__":
    unittest.main()
