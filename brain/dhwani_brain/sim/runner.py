# -*- coding: utf-8 -*-
"""One simulated mission: the world, a Brain, and a small executive between them.

The executive here is also the reference for the real ROS executive: it asks the brain for a
decision whenever a goal is reached (after a DWELL_SCAN if asked), every 10 s while driving, and
immediately on events (new or confirmed sighting, link up/down, fresh rubble, a plan that fails);
it plans on its own copy of the map (move_base's job on the rover) and never drives into unknown.

SCORE (per episode, from the task statement):
    10 x reported + 4 x found_not_reported_but_returned - 30 x lost - 0.01 x seconds
  found     a real person confirmed (>= 3 camera hits)
  reported  found, and the link was up at or after the confirmation, or the rover got home
  returned  the mission ended with the rover within 1 m of home
  lost      battery empty away from home, trapped (no route home in the TRUE map), holding
            with no route for 10 min, or still out in the mine at the 3 h cap
  With "reported" defined to include reaching home, found_not_reported_but_returned is always 0;
  the term is kept so the formula matches the statement. reported_radio counts reports that went
  out over the link before the rover was home, for anyone who wants the stricter reading.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

from ..brain import Brain
from .mapgen import generate
from .world import World, SimConfig

REDECIDE_S = 10.0
HOLD_S = 10.0
DWELL_S = 5.0
STUCK_HOLD_S = 600.0
STAY_S = 60.0


def sample_mission(rng):
    """The mission mix used for training AND evaluation (both policies see the same missions)."""
    task = "SEARCH_SURVIVORS" if rng.rand() < 0.8 else "MAP_AREA"
    m = {"task": task,
         "on_person_found": str(rng.choice(["REPORT_AND_CONTINUE", "REPORT_AND_RETURN", "STAY"],
                                           p=[0.65, 0.25, 0.10])),
         "on_link_loss": str(rng.choice(["CONTINUE_THEN_RETURN", "RETURN_NOW", "CONTINUE", "HOLD"],
                                        p=[0.55, 0.10, 0.25, 0.10])),
         "first_branch": str(rng.choice(["ANY", "LEFT", "RIGHT", "STRAIGHT"], p=[0.6, 0.15, 0.15, 0.10])),
         "return_when_done": bool(rng.rand() < 0.92),
         "max_duration_min": None if rng.rand() < 0.65 else float(rng.randint(15, 61)),
         "max_distance_m": None if rng.rand() < 0.75 else float(rng.randint(150, 801)),
         "min_battery_pct": None if rng.rand() < 0.6 else float(rng.randint(15, 41)),
         "speed": "SLOW" if rng.rand() < 0.1 else "NORMAL",
         "understood": True, "confidence": 0.95, "text": "simulated"}
    return m


def run_episode(style, seed, rockfall=False, link_loss=True, params=None, sampler=None, recorder=None,
                trace=False, mission=None, mine=None, t_max=None):
    """Run one mission. Returns a metrics dict (plus 'trace' and 'world' if trace=True).

    sampler/recorder are training hooks: sampler replaces the learned policy's argmax at
    discretionary decisions; recorder(brain.last, decision, world) is called after every decision.
    """
    ep = Episode(style, seed, rockfall, link_loss, params, mission, mine, trace, t_max)
    ep.run(sampler, recorder)
    return ep.metrics()


class Episode(object):
    """A resumable mission. run() advances until the mission ends, or until stop(ep) says so at a
    decision point (before the brain is asked), so the whole episode can be deep-copied there and
    replayed with different actions under the same random numbers (used for calibration)."""

    def __init__(self, style, seed, rockfall=False, link_loss=True, params=None, mission=None, mine=None,
                 trace=False, t_max=None):
        self.style, self.seed, self.rockfall, self.link_loss = style, seed, rockfall, link_loss
        self.mine = mine if mine is not None else generate(style, seed)
        rng = np.random.RandomState((seed * 7919 + 17) % (2 ** 31))
        self.mission_in = mission if mission is not None else sample_mission(rng)
        cfg = SimConfig(rockfall=rockfall, link_loss=link_loss)
        if t_max is not None:
            cfg.t_max = t_max
        self.cfg = cfg
        speed = cfg.slow_speed if self.mission_in.get("speed") == "SLOW" else cfg.speed
        self.world = World(self.mine, cfg, (seed * 104729 + 3) % (2 ** 31), speed=speed)
        self.brain = Brain(self.mission_in, params)
        self.mode = "decide"
        self.path, self.k, self.d = None, 0, None
        self.t_dec = 0.0
        self.dwell_left = 0.0
        self.outcome = None
        self.safety_hold_t = 0.0
        self.stay_t = 0.0
        self.n_dec = 0
        self.n_learned = 0
        self.n_disc = 0
        self.lat = []
        self.tr = {"xy": [], "decisions": [], "events": []} if trace else None
        self.fb_ok = None

    def run(self, sampler=None, recorder=None, stop=None):
        world, brain, cfg = self.world, self.brain, self.cfg
        dt = cfg.dt
        while self.outcome is None:
            if self.mode == "decide":
                if stop is not None and stop(self):
                    return False
                d = brain.decide(world.obs(), sample=sampler)
                self.d = d
                self.n_dec += 1
                self.lat.append(d.get("timing_ms", {}).get("total", 0.0))
                if d.get("tier") == "discretionary":
                    self.n_disc += 1
                if d.get("source") == "learned":
                    self.n_learned += 1
                if recorder is not None:
                    recorder(brain.last, d, world)
                if self.tr is not None:
                    self.tr["decisions"].append({"t": world.t, "x": world.x, "y": world.y, "kind": d["kind"],
                                                 "goal": d["goal"], "sensor": d["sensor"], "reason": d["reason"],
                                                 "source": d["source"], "tier": d.get("tier")})
                if self.fb_ok is None and brain.state.get("branch_committed"):
                    self.fb_ok = True
                if "error" in d:
                    raise RuntimeError(d["error"])
                if d["kind"] == "DONE":
                    self.outcome = "done"
                    break
                self.t_dec = world.t
                if d["kind"] == "HOLD" and d.get("tier") == "safety":
                    self.safety_hold_t = self.safety_hold_t or world.t
                else:
                    self.safety_hold_t = 0.0
                goal = d["goal"]
                if goal is None:
                    self.mode = "hold"
                elif math.hypot(goal[0] - world.x, goal[1] - world.y) <= 0.3:
                    self.mode = _arrive(world, d)
                    if self.mode == "decide":            # nothing to do here: let time pass
                        self.mode = "hold1"
                else:
                    # the brain's own geodesic path is valid on this exact map: follow it, and only
                    # re-plan (as move_base would) when the map closes it
                    path = [tuple(p) for p in d.get("path") or []]
                    if len(path) < 2 or math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) > 0.5:
                        path = world.plan(goal[0], goal[1])
                    self.path, self.k = path, 0
                    if not path:
                        world.blocked.append((round(goal[0], 2), round(goal[1], 2), 0.4))   # failed plan
                        world.events.append((world.t, "plan_failed", goal[0], goal[1]))
                        self.mode = "hold1"
                    else:
                        self.mode = "drive"
                continue
            self._step(dt)
        return True

    def _step(self, dt):
        world, d = self.world, self.d
        ev = False
        held = False
        if self.mode == "dwell_start":
            world.dwelling = True
            if d["goal"] is not None:
                world.yaw = d["goal"][2]
            self.dwell_left = DWELL_S
            self.mode = "dwell"
        if self.mode == "drive":
            if world.path_blocked(self.path, self.k):
                goal = d["goal"]
                self.path = world.plan(goal[0], goal[1])
                self.k = 0
                if not self.path:
                    world.events.append((world.t, "path_closed", world.x, world.y))
                    self.mode = "decide"
                    return
            self.k, moved = world.advance(self.path, self.k, world.speed * dt)
            world.tick(dt, moved)
            if self.k >= len(self.path):
                self.mode = _arrive(world, d)
            elif world.t - self.t_dec >= REDECIDE_S:
                self.mode = "decide"
            if moved < 1e-6 and self.mode == "drive":
                self.mode = "decide"                     # stuck against something: re-plan
        elif self.mode == "dwell":
            world.tick(dt, 0.0)
            self.dwell_left -= dt
            if self.dwell_left <= 1e-9:
                world.dwelling = False
                self.mode = "decide"
        elif self.mode == "hold":
            world.tick(HOLD_S, 0.0)
            ev |= world.camera(HOLD_S)
            held = True
            self.mode = "decide"
        elif self.mode == "hold1":
            world.tick(dt, 0.0)
            self.mode = "decide"

        ev |= world.reveal()
        if not held:
            ev |= world.camera(dt)
        ev |= world.update_link()
        world.maybe_rockfall()
        if self.tr is not None:
            self.tr["xy"].append((world.t, world.x, world.y, world.link_ok))
        if ev and self.mode in ("drive", "hold1"):
            self.mode = "decide"

        # ---- termination ------------------------------------------------------------------
        if world.battery <= 0.0:
            self.outcome = "battery"
        elif world.t >= self.cfg.t_max:
            self.outcome = "timeout"
        elif self.safety_hold_t and world.t - self.safety_hold_t >= STUCK_HOLD_S:
            self.outcome = "stuck_hold"
        elif d is not None and d["kind"] == "HOLD" and d.get("tier") == "mission" and \
                self.brain.mission["on_person_found"] == "STAY" and "staying with the person" in d["reason"]:
            self.stay_t = self.stay_t or world.t
            if world.t - self.stay_t >= STAY_S:
                self.outcome = "stayed"
        else:
            self.stay_t = 0.0

    def score_now(self):
        """Score if the mission ended now with the rover where it is (no return credit, no loss)."""
        w = self.world
        rep = sum(1 for p in w.persons if p["reported_t"] is not None)
        return 10.0 * rep - 0.01 * w.t

    def metrics(self):
        return _metrics(self.world, self.brain, self.mission_in, self.outcome, self.n_dec, self.n_disc,
                        self.n_learned, self.lat, self.fb_ok, self.tr, self.style, self.seed, self.rockfall,
                        self.link_loss)


def _arrive(world, d):
    if d["sensor"] == "DWELL_SCAN":
        return "dwell_start"
    if d["kind"] == "HOLD":
        return "hold"
    return "decide"


def _metrics(world, brain, mission, outcome, n_dec, n_disc, n_learned, lat, fb_ok, tr, style, seed,
             rockfall, link_loss):
    at_home = math.hypot(world.x - world.home[0], world.y - world.home[1]) <= 1.0
    returned = at_home and outcome in ("done", "timeout", "battery", "stuck_hold")
    trapped = False
    if not at_home and outcome in ("battery", "stuck_hold", "timeout", "done"):
        trapped = not world.home_connected_truth()
    if returned:
        lost = False
    elif outcome in ("battery", "stuck_hold", "timeout"):
        lost = True
    elif outcome == "done":
        lost = trapped
    else:                                            # stayed with a person
        lost = trapped
    found = [p for p in world.persons if p["found_t"] is not None]
    reported = [p for p in found if p["reported_t"] is not None or returned]
    radio = [p for p in found if p["reported_radio"]]
    fnrr = [p for p in found if p not in reported and returned]      # empty by definition, see doc
    score = 10.0 * len(reported) + 4.0 * len(fnrr) - 30.0 * float(lost) - 0.01 * world.t
    m = {"style": style, "seed": int(seed), "rockfall": bool(rockfall), "link_loss": bool(link_loss),
         "task": mission["task"], "on_link_loss": brain.mission["on_link_loss"],
         "on_person_found": brain.mission["on_person_found"], "first_branch": brain.mission["first_branch"],
         "outcome": outcome, "persons": len(world.persons), "found": len(found), "reported": len(reported),
         "reported_radio": len(radio), "returned": bool(returned), "lost": bool(lost), "trapped": bool(trapped),
         "time_s": float(world.t), "distance_m": float(world.odo), "battery_end": float(world.battery),
         "score": float(score), "decisions": n_dec, "discretionary": n_disc, "learned_used": n_learned,
         "latency_ms_median": float(np.median(lat)) if lat else 0.0,
         "rockfall_happened": world.rockfall_at is not None, "rockfall_route_left": world.rockfall_route_left,
         "link_ever_lost": bool(world.link_ever_lost),
         "blocked": len(world.blocked), "first_branch_taken": bool(fb_ok)}
    if tr is not None:
        tr["events"] = list(world.events)
        m["trace"] = tr
        m["world"] = world
    return m
