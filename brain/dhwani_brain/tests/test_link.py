# -*- coding: utf-8 -*-
"""Link loss, one test per on_link_loss option, plus the person-report interaction."""
from __future__ import division, print_function, absolute_import

import math
import unittest

from dhwani_brain import Brain
from dhwani_brain.tests.synth import Mine, obs_for


def scene(**kw):
    """Straight 60 m corridor, known to x = 25; robot at x = 20 heading east; link lost at x = 10."""
    m = Mine(60, 6, [(1, 2, 59, 4)])
    o = obs_for(m, (20.0, 3.0, 0.0), home=(2.0, 3.0), reveal=[(0, 0, 25, 6)], link_ok=False,
                since_contact_s=100.0, last_link_pose=(10.0, 3.0), battery_pct=80.0)
    o.update(kw)
    return o


def mission(oll, **kw):
    m = {"task": "SEARCH_SURVIVORS", "on_link_loss": oll}
    m.update(kw)
    return m


class TestLinkLoss(unittest.TestCase):
    def test_continue(self):
        d = Brain(mission("CONTINUE")).decide(scene(since_contact_s=5000.0))
        self.assertEqual(d["kind"], "EXPLORE", d["reason"])

    def test_continue_then_return_within_grace(self):
        d = Brain(mission("CONTINUE_THEN_RETURN")).decide(scene(since_contact_s=100.0))
        self.assertEqual(d["kind"], "EXPLORE", d["reason"])

    def test_continue_then_return_after_grace(self):
        d = Brain(mission("CONTINUE_THEN_RETURN")).decide(scene(since_contact_s=700.0))
        self.assertEqual(d["kind"], "RETURN_TO_LINK", d["reason"])
        self.assertLess(math.hypot(d["goal"][0] - 10.0, d["goal"][1] - 3.0), 0.6)

    def test_continue_then_return_skips_frontiers_it_cannot_reach_in_time(self):
        # 550 s silent: reaching the frontier and coming back to the link point would exceed 600 s
        d = Brain(mission("CONTINUE_THEN_RETURN")).decide(scene(since_contact_s=550.0))
        self.assertNotEqual(d["kind"], "EXPLORE", d["reason"])

    def test_return_now_latches(self):
        b = Brain(mission("RETURN_NOW"))
        d = b.decide(scene())
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        d = b.decide(scene(link_ok=True, since_contact_s=0.0))       # link back: still going home
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])

    def test_hold(self):
        d = Brain(mission("HOLD")).decide(scene())
        self.assertEqual(d["kind"], "HOLD", d["reason"])
        self.assertIsNone(d["goal"])

    def test_hold_yields_to_battery_reserve(self):
        d = Brain(mission("HOLD")).decide(scene(battery_pct=8.0))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])
        self.assertEqual(d["tier"], "safety")

    def test_link_ok_explores_normally(self):
        d = Brain(mission("HOLD")).decide(scene(link_ok=True, since_contact_s=0.0))
        self.assertEqual(d["kind"], "EXPLORE")

    def test_unreported_person_goes_to_link(self):
        p = [{"x": 22.0, "y": 3.0, "confirmed": True, "reported": False, "hits": 4}]
        d = Brain(mission("CONTINUE", on_person_found="REPORT_AND_CONTINUE")).decide(scene(persons=p))
        self.assertEqual(d["kind"], "RETURN_TO_LINK", d["reason"])
        d = Brain(mission("CONTINUE", on_person_found="REPORT_AND_RETURN")).decide(scene(persons=p))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])

    def test_no_link_even_at_link_point_goes_home(self):
        d = Brain(mission("CONTINUE_THEN_RETURN")).decide(
            scene(pose=(10.0, 3.0, 0.0), since_contact_s=700.0))
        self.assertEqual(d["kind"], "RETURN_HOME", d["reason"])


if __name__ == "__main__":
    unittest.main()
