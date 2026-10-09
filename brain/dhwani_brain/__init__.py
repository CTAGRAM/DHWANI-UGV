# -*- coding: utf-8 -*-
"""DHWANI on-board decision brain (pure NumPy; python 2.7 and 3.6; numpy >= 1.13).

    from dhwani_brain import Brain
    brain = Brain(mission_dict, params=None)
    decision = brain.decide(obs)

See README.md one directory up for the obs/decision contract and FEATURES.md for the features.
"""
from __future__ import division, print_function, absolute_import

from .brain import Brain
from .params import DEFAULT_PARAMS, normalize_mission

__all__ = ["Brain", "DEFAULT_PARAMS", "normalize_mission"]
__version__ = "1.0.0"
