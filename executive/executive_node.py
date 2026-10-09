#!/usr/bin/env python
"""executive_node.py: the rover carries out a mission by itself, with or without a link.

WHAT IT DOES. It takes a mission (the JSON the plain-English parser produces, on /mission/plan),
and once an operator ARMS it, it keeps asking the on-board brain (dhwani_brain.Brain) what to do
next: explore a frontier, go closer to check a person, go home, go back to where the link last
worked, or hold. It turns each decision into a move_base goal, watches the goal, and asks again
when the goal ends, when something changes (a person seen, the link lost or back, battery low, a
route blocked) or every 20 s in any case.

WHY IT LIVES ON THE ROVER. The base station is exactly what disappears underground. Every input
here is local: the SLAM map, TF, odometry, the IMU, the camera topics. The link is only an input
("is anyone listening?"), never a requirement. What happens when it drops is the mission's own
on_link_loss setting, decided by the brain, not a timeout that strands the rover.

SAFETY, which overrides the brain every time:
  * Nothing moves unless ALL of: ~live is true in ~/dhwani_executive.json, a mission is loaded,
    and an operator has sent /executive/arm True (the dashboard asks for the PIN and a confirm).
  * STOP on /mission/command, /executive/arm False, a tilt beyond the limit, or a cliff flag from
    the depth camera cancels the goal and publishes zero velocity at once.
  * A goal that fails, or a rover that stops moving while it should be moving, marks the spot as
    blocked so the brain plans around it (this is how a rockfall behind the rover is handled when
    the map has not caught up yet; once gmapping adds the rubble, the map itself blocks it).
  * With ~live false (the default) it runs every step except sending goals: TEST MODE. The
    dashboard shows the decisions, the reasons and the options it weighed.

Python 2.7, like mission_node.py: tf and actionlib are needed and exist only for 2.7 on Melodic.
"""
from __future__ import division, print_function

import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import rospy
import tf
import actionlib
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Imu
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalID, GoalStatus

HOME_DIR = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME_DIR, "dhwani_brain"))
BRAIN_ERR = None
try:
    from dhwani_brain import Brain
except Exception as e:                      # the node still runs and says why it cannot decide
    Brain = None
    BRAIN_ERR = repr(e)

CONFIG = os.path.join(HOME_DIR, "dhwani_executive.json")
DEFAULTS = {
    "live": False,               # False = TEST MODE: decide and report, never send goals
    "use_learned": False,        # learned policy advises only when true AND confident AND safe
    "battery_start_pct": 100.0,  # estimate; the dashboard can set the real figure at start
    "drain_pct_per_min": 0.72,   # estimate (= the brain's 0.012 %/s); replace with a real battery log
    "drain_pct_per_m": 0.04,     # estimate (= the brain's default); replace with a real battery log
    "tilt_limit_deg": 25.0,
    "stuck_s": 6.0,
    "link_timeout_s": 5.0,
    "decide_every_s": 10.0,
    "max_goal_s": 150.0,
    "dwell_s": 5.0,
}
PERSON_MERGE_M = 1.0
CLIFF_CONFIRM = 2          # 30 Sep: /oak/cliff frames in a row before acting; shiny tiles give 1-frame blips
# 30 Sep: the real battery. The expansion board's INA219 (I2C bus 1, 0x41, factory config 0x399F)
# gives the pack voltage in its bus-voltage register; it is read every BATTERY_READ_S, smoothed over
# about BATTERY_TAU_S (motor bursts sag it for seconds), and turned into charge left with a typical
# Li-ion 18650 curve per cell. The curve is NOT calibrated on these cells; 0 % sits at 3.4 V a cell
# to keep a margin above the pack protection's cut-off. Without a reading the time model continues.
BATTERY_BUS, BATTERY_ADDR = "1", "0x41"
BATTERY_READ_S = 5.0
BATTERY_TAU_S = 30.0
CELLS = 3
CELL_CURVE = ((3.40, 0.0), (3.50, 12.0), (3.60, 25.0), (3.70, 40.0), (3.80, 55.0), (3.90, 68.0),
              (4.00, 80.0), (4.10, 90.0), (4.20, 100.0))
BLOCK_CLEAR_M = 0.4        # 30 Sep: a no-go disc's edge stays this far from the rover. The brain paints it
                           # 0.25 m wider, and a disc over the rover's own cell walls the rover in
PERSON_CONFIRM_HITS = 3


def read_pack_volts():
    """Pack voltage from the INA219 bus-voltage register (4 mV a step), one combined I2C transfer so
    another reader cannot move the register pointer in between. Raises on any failure."""
    out = subprocess.check_output(["i2ctransfer", "-y", BATTERY_BUS, "w1@" + BATTERY_ADDR, "0x02", "r2"],
                                  stderr=subprocess.STDOUT)
    hi, lo = [int(b, 16) for b in out.split()]
    raw = (hi << 8) | lo
    if raw & 1:
        raise ValueError("INA219 bus voltage overflow")
    return (raw >> 3) * 0.004


def pack_pct(volts):
    """Charge left (%) from the pack voltage, by CELL_CURVE."""
    v = volts / CELLS
    if v <= CELL_CURVE[0][0]:
        return 0.0
    for (v0, p0), (v1, p1) in zip(CELL_CURVE, CELL_CURVE[1:]):
        if v <= v1:
            return p0 + (p1 - p0) * (v - v0) / (v1 - v0)
    return 100.0


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG) as f:
            cfg.update(json.load(f))
    except (IOError, ValueError):
        pass
    return cfg


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Executive(object):
    def __init__(self):
        self.cfg = load_config()
        self.live = bool(self.cfg["live"])
        self.lock = threading.RLock()
        self.tfl = tf.TransformListener()

        self.mission = None
        self.phase = "IDLE"           # IDLE READY ACTIVE DWELL HOLD HALTED DONE
        self.armed = False
        self.brain = None
        self.decision = None
        self.reason = "Waiting for a mission."
        self.decide_ms = None
        self.last_decide = 0.0
        self.event_flag = False

        self.grid = None
        self.grid_info = None
        self.odom_v = 0.0
        self.odo_m = 0.0
        self.last_xy = None
        self.tilt_deg = 0.0
        self.cliff = False
        self.cliff_run = 0
        self.persons = []
        self.blocked = []
        self.trail = []
        self.home = None
        self.t_start = None
        self.last_link = 0.0
        self.link_was_ok = False
        self.last_link_pose = None
        self.battery_start = float(self.cfg["battery_start_pct"])
        self.battery_t0 = time.time()
        self.odo_at_batt = 0.0
        self.batt_v = None             # smoothed pack voltage, None until the INA219 has answered
        self.batt_read_at = 0.0
        self.batt_fails = 0

        self.goal = None               # dict: kind, xy, sent_at, sensor
        self.goal_active = False
        self.dwell_until = 0.0
        self.slow_since = None
        self.last_now = None
        self.nav_warned = 0.0

        self.pub_state = rospy.Publisher("/executive/state", String, queue_size=2)
        self.pub_events = rospy.Publisher("/events", String, queue_size=10)
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pub_cancel = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self.mb = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.mb_ok = self.mb.wait_for_server(rospy.Duration(10.0))
        if not self.mb_ok:
            rospy.logwarn("move_base action server not found; goals cannot be sent")

        rospy.Subscriber("/mission/plan", String, self.on_plan, queue_size=2)
        rospy.Subscriber("/executive/arm", Bool, self.on_arm, queue_size=2)
        rospy.Subscriber("/executive/battery", Float32, self.on_battery, queue_size=2)
        rospy.Subscriber("/mission/command", String, self.on_command, queue_size=5)
        rospy.Subscriber("/mission/permit", Bool, self.on_link, queue_size=5)
        rospy.Subscriber("/dashboard/heartbeat", String, self.on_link, queue_size=5)
        rospy.Subscriber("/map", numpy_msg(OccupancyGrid), self.on_map, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber("/odom", Odometry, self.on_odom, queue_size=5)
        rospy.Subscriber("/imu", Imu, self.on_imu, queue_size=5)
        rospy.Subscriber("/oak/person", PointStamped, self.on_person, queue_size=10)
        rospy.Subscriber("/oak/cliff", Bool, self.on_cliff, queue_size=5)
        self.mission_state = "?"
        rospy.Subscriber("/mission/state", String, self.on_mstate, queue_size=5)

    # ------------------------------------------------------------------ inputs
    def on_plan(self, msg):
        try:
            m = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            task = m.get("task")
            if task == "HALT":
                self.halt("mission HALT")
                return
            if task in (None, "UNSUPPORTED") or not m.get("understood", True):
                self.event("REFUSED", "mission not understood")
                return
            self.mission = m
            # 30 Sep: marks from an earlier mission (often a false drop on shiny tiles) must not wall
            # in the next one; a real drop is seen and marked again when the rover comes back to it.
            self.blocked = []
            self.brain = None
            if Brain is not None:
                try:
                    self.brain = Brain(m, self.brain_params())
                except Exception as e:
                    self.reason = "brain failed to start: %s" % e
            self.decision = None
            self.phase = "ACTIVE" if self.armed else "READY"
            self.reason = "Mission loaded: %s. %s" % (task, "" if self.armed else "Waiting for ARM.")
            self.event_flag = True
            self.event("MISSION", "%s loaded%s" % (task, "" if self.live else " (test mode)"))

    def on_arm(self, msg):
        with self.lock:
            if not msg.data:
                if self.armed:
                    self.halt("disarmed by operator", phase="READY" if self.mission else "IDLE")
                return
            if self.mission is None:
                self.reason = "ARM ignored: no mission loaded."
                return
            if self.mission_state.startswith("DRIVING"):
                self.reason = "ARM refused: mission_node is driving a waypoint route."
                return
            self.armed = True
            self.phase = "ACTIVE"
            pose = self.pose()
            if self.home is None and pose is not None:
                self.home = (pose[0], pose[1])
            self.t_start = time.time()
            self.event_flag = True
            self.event("ARMED", "rover is deciding by itself%s" % ("" if self.live else " (test mode: no motion)"))

    def on_battery(self, msg):
        with self.lock:
            self.battery_start = float(msg.data)
            self.battery_t0 = time.time()
            self.odo_at_batt = self.odo_m

    def on_command(self, msg):
        if msg.data.strip().upper().startswith("STOP"):
            self.halt("operator STOP")

    def on_link(self, msg):
        self.last_link = time.time()

    def on_mstate(self, msg):
        self.mission_state = msg.data

    def on_map(self, m):
        # numpy_msg delivers data as an int8 array: 4.4 ms instead of 481 ms for the tuple on the Nano
        g = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        with self.lock:
            self.grid = g
            self.grid_info = (m.info.resolution, m.info.origin.position.x, m.info.origin.position.y)

    def on_odom(self, o):
        self.odom_v = o.twist.twist.linear.x

    def on_imu(self, m):
        q = m.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.tilt_deg = math.degrees(max(abs(roll), abs(pitch)))

    def mark_blocked(self, x, y, r, pose):
        """Tell the brain not to plan through a disc. 30 Sep: marks 0.5-0.6 m ahead with r 0.45-0.5
        reached back over the rover itself; two of them left it one reachable cell and the brain
        reported the room explored. The radius is cut so the disc stays clear of the rover."""
        if pose is not None:
            r = min(r, math.hypot(x - pose[0], y - pose[1]) - BLOCK_CLEAR_M)
        if r >= 0.1:
            self.blocked.append((x, y, r))

    def on_cliff(self, m):
        self.cliff_run = self.cliff_run + 1 if m.data else 0
        if self.cliff_run >= CLIFF_CONFIRM and not self.cliff:
            self.cliff = True
            with self.lock:
                if self.goal_active:
                    self.cancel_goal()
                    self.zero()
                    p = self.pose()
                    if p:
                        self.mark_blocked(p[0] + 0.6 * math.cos(p[2]), p[1] + 0.6 * math.sin(p[2]), 0.5, p)
                    self.event("CLIFF", "drop-off ahead, marked and avoided")
                    self.event_flag = True
        elif not m.data:
            self.cliff = False

    def on_person(self, msg):
        try:
            self.tfl.waitForTransform("map", msg.header.frame_id or "base_footprint", rospy.Time(0), rospy.Duration(0.3))
            m = msg
            m.header.stamp = rospy.Time(0)
            p = self.tfl.transformPoint("map", m).point
        except Exception:
            return
        with self.lock:
            for q in self.persons:
                if math.hypot(q["x"] - p.x, q["y"] - p.y) < PERSON_MERGE_M:
                    q["hits"] += 1
                    q["x"] = 0.8 * q["x"] + 0.2 * p.x
                    q["y"] = 0.8 * q["y"] + 0.2 * p.y
                    if not q["confirmed"] and q["hits"] >= PERSON_CONFIRM_HITS:
                        q["confirmed"] = True
                        self.event("PERSON", "person confirmed at (%.1f, %.1f)" % (q["x"], q["y"]), (q["x"], q["y"]))
                        self.event_flag = True
                    return
            self.persons.append({"x": p.x, "y": p.y, "hits": 1, "confirmed": False, "reported": False})
            self.event_flag = True

    # ------------------------------------------------------------------ helpers
    def brain_params(self):
        p = {"use_learned": bool(self.cfg["use_learned"]),
             "drain_pct_per_m": float(self.cfg["drain_pct_per_m"]),
             "drain_pct_per_s": float(self.cfg["drain_pct_per_min"]) / 60.0,
             "dwell_s": float(self.cfg["dwell_s"])}
        try:
            with open(os.path.join(HOME_DIR, "waypoints.json")) as f:
                p["waypoints"] = json.load(f)
        except (IOError, ValueError):
            pass
        return p

    def pose(self):
        try:
            (t, q) = self.tfl.lookupTransform("map", "base_footprint", rospy.Time(0))
        except Exception:
            return None
        yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
        return (t[0], t[1], yaw)

    def battery(self):
        mins = (time.time() - self.battery_t0) / 60.0
        used = mins * self.cfg["drain_pct_per_min"] + (self.odo_m - self.odo_at_batt) * self.cfg["drain_pct_per_m"]
        return max(0.0, self.battery_start - used)

    def link_ok(self):
        return (time.time() - self.last_link) < self.cfg["link_timeout_s"]

    def event(self, kind, text, xy=None):
        e = {"t": round(time.time(), 1), "kind": kind, "text": text, "src": "executive"}
        if xy:
            e["x"], e["y"] = round(xy[0], 2), round(xy[1], 2)
        self.pub_events.publish(String(json.dumps(e)))
        rospy.loginfo("EXEC %s: %s", kind, text)

    def zero(self):
        if not self.live:                   # TEST MODE never touches /cmd_vel
            return
        for _ in range(5):
            self.pub_cmd.publish(Twist())

    def cancel_goal(self):
        if self.goal_active:
            try:
                self.mb.cancel_all_goals()
            except Exception:
                pass
        if self.live:                       # TEST MODE never touches move_base
            self.pub_cancel.publish(GoalID())
        self.goal_active = False

    def halt(self, why, phase="HALTED"):
        with self.lock:
            was = self.armed or self.goal_active
            self.cancel_goal()
            self.zero()
            self.armed = False
            self.phase = phase if self.mission else "IDLE"
            self.goal = None
            self.reason = "Stopped: %s." % why
            if was:
                self.event("HALT", why)

    def obs(self, pose):
        res, ox, oy = self.grid_info
        return {
            "grid": self.grid, "res": res, "origin": (ox, oy),
            "pose": pose, "home": self.home or (pose[0], pose[1]),
            "battery_pct": self.battery(), "link_ok": self.link_ok(),
            "since_contact_s": time.time() - self.last_link if self.last_link else 1e6,
            "elapsed_s": time.time() - self.t_start if self.t_start else 0.0,
            "odo_m": self.odo_m,
            "persons": [dict(p) for p in self.persons],
            "blocked": list(self.blocked),
            "last_link_pose": self.last_link_pose,
            "trail": self.trail[-200:],
        }

    # ------------------------------------------------------------------ the loop
    def read_battery(self, now):
        """Every BATTERY_READ_S: the measured pack voltage re-bases the battery estimate. Called outside
        the lock (it runs a short subprocess)."""
        if now - self.batt_read_at < BATTERY_READ_S:
            return
        self.batt_read_at = now
        try:
            v = read_pack_volts()
            if not 6.0 < v < 14.0:
                raise ValueError("implausible pack voltage %.2f V" % v)
        except Exception as e:
            self.batt_fails += 1
            if self.batt_fails == 3:
                rospy.logwarn("battery: INA219 not answering (%s); using the time estimate", e)
            return
        with self.lock:
            self.batt_fails = 0
            a = 1.0 if self.batt_v is None else min(1.0, BATTERY_READ_S / BATTERY_TAU_S)
            self.batt_v = v if self.batt_v is None else self.batt_v + a * (v - self.batt_v)
            self.battery_start, self.battery_t0, self.odo_at_batt = pack_pct(self.batt_v), now, self.odo_m

    def step(self):
        pose = self.pose()
        now = time.time()
        self.read_battery(now)
        with self.lock:
            # 30 Sep: the Jetson has no clock battery. It boots at the last shutdown time and jumps
            # when network time arrives (+44 h on 30 Sep); the battery estimate took the jump for
            # 44 h of running (0 %, every mission ended the moment it was armed). This loop runs at
            # 2 Hz, so a gap over 60 s is a clock jump: move the times stored before it (a heartbeat
            # that arrived after the jump is already right and must not move 44 h into the future).
            if self.last_now is not None and abs(now - self.last_now) > 60.0:
                j, before = now - self.last_now, self.last_now
                old = lambda t: bool(t) and abs(t - before) < abs(t - now)
                for k in ("battery_t0", "t_start", "slow_since", "last_link", "last_decide", "dwell_until", "nav_warned",
                          "batt_read_at"):
                    if old(getattr(self, k)):
                        setattr(self, k, getattr(self, k) + j)
                if self.goal and old(self.goal["sent_at"]):
                    self.goal["sent_at"] += j
                rospy.logwarn("clock jumped %+.0f s; stored times moved with it", j)
            self.last_now = now
            if pose is not None:
                if self.last_xy is not None:
                    d = math.hypot(pose[0] - self.last_xy[0], pose[1] - self.last_xy[1])
                    if d < 2.0:
                        self.odo_m += d
                self.last_xy = (pose[0], pose[1])
                if not self.trail or math.hypot(pose[0] - self.trail[-1][0], pose[1] - self.trail[-1][1]) > 0.5:
                    self.trail.append((round(pose[0], 2), round(pose[1], 2)))
            ok = self.link_ok()
            if ok and pose is not None:
                self.last_link_pose = (pose[0], pose[1])
                for p in self.persons:
                    if p["confirmed"] and not p["reported"]:
                        p["reported"] = True
                        self.event("REPORTED", "person at (%.1f, %.1f) reported to base" % (p["x"], p["y"]), (p["x"], p["y"]))
            if ok != self.link_was_ok:
                self.link_was_ok = ok
                self.event_flag = True
                if self.armed:
                    self.event("LINK", "link %s" % ("back" if ok else "lost: continuing by mission rule"))

            # safety that overrides the brain
            if self.armed and self.tilt_deg > self.cfg["tilt_limit_deg"]:
                self.halt("tilt %.0f deg exceeds %.0f" % (self.tilt_deg, self.cfg["tilt_limit_deg"]))
            if self.armed and self.phase == "ACTIVE" and self.goal_active and self.live:
                if abs(self.odom_v) < 0.02:
                    self.slow_since = self.slow_since or now
                    if now - self.slow_since > self.cfg["stuck_s"]:
                        self.cancel_goal()
                        if pose:
                            self.mark_blocked(pose[0] + 0.5 * math.cos(pose[2]), pose[1] + 0.5 * math.sin(pose[2]), 0.45, pose)
                        self.event("STUCK", "no progress for %.0f s, marking the way ahead blocked" % self.cfg["stuck_s"])
                        self.slow_since = None
                        self.event_flag = True
                else:
                    self.slow_since = None

            # goal bookkeeping
            if self.goal_active and self.live:
                st = self.mb.get_state()
                if st == GoalStatus.SUCCEEDED:
                    self.goal_active = False
                    if self.goal and self.goal.get("sensor") == "DWELL_SCAN":
                        self.dwell_until = now + self.cfg["dwell_s"]
                        self.phase = "DWELL"
                    self.event_flag = True
                elif st in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST):
                    self.goal_active = False
                    if self.goal and self.goal.get("xy"):
                        gx, gy = self.goal["xy"]
                        self.mark_blocked(gx, gy, 0.4, pose)
                    self.event("FAILED", "move_base could not reach the goal; it is marked blocked")
                    self.event_flag = True
                elif self.goal and now - self.goal["sent_at"] > self.cfg["max_goal_s"]:
                    self.cancel_goal()
                    self.event_flag = True
            if self.phase == "DWELL" and now >= self.dwell_until:
                self.phase = "ACTIVE"
                self.event_flag = True

            # decide
            if self.armed and self.phase == "ACTIVE" and pose is not None and self.grid is not None:
                due = (not self.goal_active and not self.live) and (now - self.last_decide > 5.0)
                due = due or (self.live and not self.goal_active)
                due = due or self.event_flag or (now - self.last_decide > self.cfg["decide_every_s"])
                if due:
                    self.decide_and_act(pose, now)
        self.publish_state(pose)

    def decide_and_act(self, pose, now):
        self.event_flag = False
        self.last_decide = now
        if self.brain is None:
            self.reason = "No decision brain installed (%s). Holding." % (BRAIN_ERR or "not loaded")
            self.decision = {"kind": "HOLD", "goal": None, "reason": self.reason, "source": "none"}
            return
        t0 = time.time()
        try:
            d = self.brain.decide(self.obs(pose))
        except Exception as e:
            self.reason = "brain error: %s. Holding." % e
            self.cancel_goal()
            self.decision = {"kind": "HOLD", "goal": None, "reason": self.reason, "source": "error"}
            return
        self.decide_ms = round((time.time() - t0) * 1000.0, 2)
        self.decision = d
        self.reason = d.get("reason", "")
        kind = d.get("kind")
        if kind in ("HOLD", "DONE") and not (kind == "HOLD" and d.get("goal")):
            self.cancel_goal()
            if kind == "DONE":
                self.armed = False
                self.phase = "DONE"
                self.event("DONE", d.get("reason", "mission complete"))
            else:
                self.phase = "HOLD" if not self.live else "ACTIVE"
                self.zero()
            return
        goal = d.get("goal")
        if not goal:
            return
        same = self.goal and self.goal.get("xy") and math.hypot(goal[0] - self.goal["xy"][0], goal[1] - self.goal["xy"][1]) < 0.3
        if same and self.goal_active:
            return
        self.goal = {"kind": kind, "xy": (goal[0], goal[1]), "sent_at": now, "sensor": d.get("sensor", "MOVE")}
        if self.live and not self.mb_ok:
            # 30 Sep: at boot move_base came up about a minute after this node, the one 10 s wait at
            # start failed, and no goal was ever sent. Look again whenever a goal is due.
            self.mb_ok = self.mb.wait_for_server(rospy.Duration(0.2))
            if self.mb_ok:
                rospy.loginfo("move_base action server found")
            elif now - self.nav_warned > 30.0:
                self.nav_warned = now
                self.event("NAV DOWN", "move_base is not running, so the rover cannot drive to its goal")
        if self.live and self.mb_ok:
            g = MoveBaseGoal()
            g.target_pose.header.frame_id = "map"
            g.target_pose.header.stamp = rospy.Time.now()
            g.target_pose.pose.position.x, g.target_pose.pose.position.y = goal[0], goal[1]
            yaw = goal[2] if len(goal) > 2 and goal[2] is not None else math.atan2(goal[1] - pose[1], goal[0] - pose[0])
            g.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
            g.target_pose.pose.orientation.w = math.cos(yaw / 2.0)
            self.mb.send_goal(g)
            self.goal_active = True
            self.slow_since = None
        sig = (kind, round(goal[0], 1), round(goal[1], 1))
        if sig != getattr(self, "_last_sig", None):     # one event per new decision, not per re-check
            self._last_sig = sig
            self.event("DECISION", "%s -> (%.1f, %.1f): %s" % (kind, goal[0], goal[1], d.get("reason", "")), (goal[0], goal[1]))

    def publish_state(self, pose):
        d = self.decision or {}
        cands = []
        for c in (d.get("candidates") or [])[:8]:
            g = c.get("goal")
            cands.append({"kind": c.get("kind"), "goal": [round(g[0], 2), round(g[1], 2)] if g else None,
                          "score": round(float(c.get("score", 0.0)), 3) if c.get("score") is not None else None})
        path = d.get("path") or []
        step = max(1, len(path) // 60)
        dec = {"kind": d.get("kind"), "sensor": d.get("sensor"), "reason": d.get("reason"),
               "source": d.get("source"), "confidence": d.get("confidence"), "tier": d.get("tier"),
               "goal": [round(d["goal"][0], 2), round(d["goal"][1], 2)] if d.get("goal") else None,
               "path": [[round(q[0], 2), round(q[1], 2)] for q in path[::step]],
               "status": d.get("status")}
        st = {"phase": self.phase, "armed": self.armed, "live": self.live, "dry_run": not self.live,
              "mission": self.mission, "decision": dec, "reason": self.reason, "candidates": cands,
              "decide_ms": self.decide_ms, "battery_pct": round(self.battery(), 1), "link_ok": self.link_ok(),
              "battery_v": round(self.batt_v, 2) if self.batt_v is not None and self.batt_fails < 3 else None,
              "home": [round(self.home[0], 2), round(self.home[1], 2)] if self.home else None,
              "persons": [{"x": round(p["x"], 2), "y": round(p["y"], 2), "confirmed": p["confirmed"],
                           "reported": p["reported"]} for p in self.persons],
              "blocked": [[round(b[0], 2), round(b[1], 2), b[2]] for b in self.blocked[-20:]],
              "brain": "ok" if Brain is not None else (BRAIN_ERR or "missing"),
              "odo_m": round(self.odo_m, 1)}
        self.pub_state.publish(String(json.dumps(st)))

    def spin(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            try:
                self.step()
            except Exception as e:
                rospy.logerr("executive step failed: %s", e)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("dhwani_executive", anonymous=True)
    ex = None
    try:
        ex = Executive()
        rospy.loginfo("executive up: live=%s brain=%s", ex.live, "ok" if Brain else BRAIN_ERR)
        ex.spin()
    finally:
        if ex is not None and ex.goal_active:
            ex.cancel_goal()
            ex.zero()
