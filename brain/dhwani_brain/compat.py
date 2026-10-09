# -*- coding: utf-8 -*-
"""Python 2.7 / 3.6 shims. Only what this package actually uses.

The brain is imported by a ROS executive under python2.7 (rospy + tf) on the rover, and the
same files run under python3 on the Mac for training and tests. Everything version-specific
lives here so the rest of the package reads as plain Python.
"""
from __future__ import division, print_function, absolute_import

import math
import sys
import time

PY2 = sys.version_info[0] == 2

# perf_counter does not exist on 2.7; time.time() has microsecond resolution on Linux.
clock = getattr(time, "perf_counter", time.time)

INF = float("inf")


def upper(x, default):
    """Normalise a mission enum field. json.loads gives unicode on py2; None means default."""
    if x is None:
        return default
    try:
        s = str(x)
    except UnicodeEncodeError:          # py2 unicode with non-ASCII: keep the ASCII part
        s = x.encode("ascii", "ignore")
    s = s.strip().upper()
    return s if s else default


def fnum(x, default=None):
    """float(x), or default when x is None / not a finite number."""
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (INF, -INF):
        return default
    return v


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
