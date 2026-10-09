# -*- coding: utf-8 -*-
"""Tiny synthetic maps for the tests: corridors drawn as rectangles, partially revealed."""
from __future__ import division, print_function, absolute_import

import numpy as np


class Mine(object):
    """Ground truth from free rectangles (x0, y0, x1, y1) in metres; origin at (0, 0)."""

    def __init__(self, w_m, h_m, rects, res=0.1):
        self.res = res
        self.H = int(round(h_m / res))
        self.W = int(round(w_m / res))
        self.free = np.zeros((self.H, self.W), dtype=bool)
        for r in rects:
            self.carve(*r)

    def carve(self, x0, y0, x1, y1, value=True):
        r = self.res
        self.free[int(round(y0 / r)):int(round(y1 / r)), int(round(x0 / r)):int(round(x1 / r))] = value

    def grid(self, reveal=None):
        """int8 occupancy: revealed free -> 0, revealed rock -> 100 (walls seen from free space),
        everything else -1. reveal: list of rectangles in metres; None = everything."""
        seen = np.zeros_like(self.free)
        if reveal is None:
            seen[:] = True
        else:
            r = self.res
            for (x0, y0, x1, y1) in reveal:
                seen[int(round(y0 / r)):int(round(y1 / r)), int(round(x0 / r)):int(round(x1 / r))] = True
        vis_free = seen & self.free
        # walls: rock cells within 2 cells of a seen free cell
        near = vis_free.copy()
        for _ in range(2):
            n = near.copy()
            n[1:, :] |= near[:-1, :]
            n[:-1, :] |= near[1:, :]
            n[:, 1:] |= near[:, :-1]
            n[:, :-1] |= near[:, 1:]
            near = n
        g = np.full(self.free.shape, -1, dtype=np.int8)
        g[vis_free] = 0
        g[near & ~self.free] = 100
        return g


def obs_for(mine, pose, home=None, reveal=None, **kw):
    """An obs dict. By default the trail is the straight line home -> robot (the robot drove up
    the corridor it stands in), so camera coverage matches what such a robot would have seen."""
    home = home if home is not None else (pose[0], pose[1])
    n = int(np.hypot(pose[0] - home[0], pose[1] - home[1]) / 0.5) + 1
    trail = [(home[0] + (pose[0] - home[0]) * t, home[1] + (pose[1] - home[1]) * t)
             for t in np.linspace(0.0, 1.0, n)]
    o = {"grid": mine.grid(reveal), "res": mine.res, "origin": (0.0, 0.0), "pose": pose,
         "home": home, "battery_pct": 90.0,
         "link_ok": True, "since_contact_s": 0.0, "elapsed_s": 60.0, "odo_m": 10.0,
         "persons": [], "blocked": [], "last_link_pose": None, "trail": trail}
    o.update(kw)
    return o


def t_junction(res=0.1):
    """Stem from (10, 2) north to a T at y=22; arms west to x=2 and east to x=18. 2 m wide."""
    return Mine(20, 30, [(9, 1, 11, 23), (1, 21, 19, 23)], res)


def cross_junction(res=0.1):
    """Stem north to a + at y=20; arms west, east and north. 2 m wide."""
    return Mine(24, 34, [(11, 1, 13, 33), (1, 19, 23, 21)], res)


def loop_mine(res=0.1):
    """Home at (3, 3). Corridor east along y=3 to x=27, north along x=27 to y=27, west along
    y=27 back to x=3, south along x=3 to home: a square loop of 2 m corridors."""
    return Mine(30, 30, [(2, 2, 28, 4), (26, 2, 28, 28), (2, 26, 28, 28), (2, 2, 4, 28)], res)


def dead_end_line(res=0.1):
    """One straight corridor from home (2, 3) east to x=38; nothing else."""
    return Mine(40, 6, [(1, 2, 39, 4)], res)
