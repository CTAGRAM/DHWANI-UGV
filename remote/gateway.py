#!/usr/bin/env python3
"""DHWANI remote gateway: the web dashboard, its live data and remote control, on one port.

WHY ONE PROCESS ON ONE PORT. A Cloudflare quick tunnel exposes exactly one local origin, and a
mentor 40 km away should only ever need one link. So this serves the page itself, pushes live
data over a WebSocket on /ws, and accepts commands on the same socket. rosbridge (:9090) stays
private: publishing /cmd_vel through a raw public rosbridge would let anyone with the link drive
the robot, with no PIN and no dead-man.

SAFETY MODEL, in order of importance.
  1. STOP is open to everyone with the link and never needs a PIN. Stopping is always safe.
  2. Driving and starting missions need the PIN, and only one person controls at a time.
  3. Driving is hold-to-drive. The browser repeats the command 10 times a second; if a command is
     more than 0.5 s old the gateway sends zero velocity and stops publishing. A dropped tab, a
     locked phone or a dead tunnel therefore stops the rover within half a second.
  4. Speed is capped here, not in the browser, so a modified page cannot exceed it.
  5. --dry-run sends every motion topic to a *_dryrun twin instead, so the whole chain can be
     tested end to end on a parked rover. The page shows a TEST MODE banner.
  6. Every command is written to logs/commands.log with the time and the sender.

WHAT IT PUBLISHES (live mode): /cmd_vel (manual drive only), /mission/command (STOP, HOME, GOTO),
/mission/permit (while a controller is connected, the same dead-man stream base_station.py sends),
/mission/plan (mission JSON for the executive), /executive/arm, /move_base/cancel,
/dashboard/heartbeat.

Runs under the rover's python3.6 with rospy from Melodic's python2.7 path, like oak_node.py.
    python3 gateway.py [--dry-run] [--port 8088]
"""
import argparse
import asyncio
import base64
import collections
import hashlib
import io
import json
import math
import os
import random
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pylib"))
# pylib carries numpy 1.19.5 with its own OpenBLAS: 2.5x the system numpy's matrix speed on the
# Nano, which makes the TinyBERT parser 45 ms instead of 100 ms. Two settings are mandatory:
# ARMV8 core type (the wheel's auto-detect crashes the Cortex-A57 with "Illegal instruction") and
# ONE thread (four threads fight ROS for the CPU and were 3x slower, measured 25 Sep 2026).
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import websockets
from http import HTTPStatus
import signal

import rospy
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan, Image
from tf2_msgs.msg import TFMessage
from actionlib_msgs.msg import GoalID

VERSION = "2026-09-28"
MAX_V = 0.30            # m/s, manual drive cap
MAX_W = 1.20            # rad/s
DRIVE_STALE_S = 0.5     # hold-to-drive: older commands are ignored and the rover is stopped
CONTROL_STALE_S = 15.0  # a controller silent this long loses the lock
TOKEN_TTL_S = 12 * 3600
CAM_FPS = 4.0
OBJ_PORT = 5605         # every OAK detection (all 80 COCO classes) from oak_v3.py, ~10 Hz
SEND_WAIT_S = 1.5       # longest a send to one viewer may wait for its socket
SLOW_BUF = 64 * 1024    # bytes queued to a viewer (websockets' own high-water mark) before it skips frequent updates
CRITICAL_MSGS = ("control", "ack", "toast", "event", "exec", "mission_state", "parsed", "auth", "hello")
CAM_W, CAM_H = 320, 240
GUARD_CORRIDOR = 0.12   # m, half width of the path checked for drops and obstacles (rover 0.16 m wide)
GUARD_STOP = 0.60       # m from the rover centre: forward driving is refused inside this
GUARD_STALE_S = 0.5     # older depth scans do not count as protection
GUARD_HOLD_S = 0.4      # a close reading keeps blocking this long: an edge that flickers still blocks
DEPTH_NEAR_MM, DEPTH_FAR_MM = 200, 5000   # depth colour scale: blue at 0.2 m to red at 5 m, like Foxglove


# Turbo colour map (Google LLC, Apache License 2.0), the same scale Foxglove uses for depth:
# 256 RGB entries, index 0 = near (dark blue) ... 255 = far (dark red).
DEPTH_LUT = np.frombuffer(bytes.fromhex(
    "30123b32154333184a341b51351e5836215f37246638276d392a733a2d793b2f803c32863d358b3e38913f3b973f3e9c"
    "4040a24143a74146ac4249b1424bb5434eba4451bf4454c34456c74559cb455ccf455ed34661d64664da4666dd4669e0"
    "466be3476ee64771e94773eb4776ee4778f0477bf2467df44680f64682f84685fa4687fb458afc458cfd448ffe4391fe"
    "4294ff4196ff4099ff3e9bfe3d9efe3ba0fd3aa3fc38a5fb37a8fa35abf833adf731aff52fb2f42eb4f22cb7f02ab9ee"
    "28bceb27bee925c0e723c3e422c5e220c7df1fc9dd1ecbda1ccdd81bd0d51ad2d21ad4d019d5cd18d7ca18d9c818dbc5"
    "18ddc218dec018e0bd19e2bb19e3b91ae4b61ce6b41de7b21fe9af20eaac22ebaa25eca727eea42aefa12cf09e2ff19b"
    "32f29835f39438f4913cf58e3ff68a43f78746f8844af8804ef97d52fa7a55fa7659fb735dfc6f61fc6c65fd6969fd66"
    "6dfe6271fe5f75fe5c79fe597dff5680ff5384ff5188ff4e8bff4b8fff4992ff4796fe4499fe429cfe409ffd3fa1fd3d"
    "a4fc3ca7fc3aa9fb39acfb38affa37b1f936b4f836b7f735b9f635bcf534bef434c1f334c3f134c6f034c8ef34cbed34"
    "cdec34d0ea34d2e935d4e735d7e535d9e436dbe236dde037dfdf37e1dd37e3db38e5d938e7d739e9d539ebd339ecd13a"
    "eecf3aefcd3af1cb3af2c93af4c73af5c53af6c33af7c13af8be39f9bc39faba39fbb838fbb637fcb336fcb136fdae35"
    "fdac34fea933fea732fea431fea130fe9e2ffe9b2dfe992cfe962bfe932afe9029fd8d27fd8a26fc8725fc8423fb8122"
    "fb7e21fa7b1ff9781ef9751df8721cf76f1af66c19f56918f46617f36315f26014f15d13f05b12ef5811ed5510ec530f"
    "eb500eea4e0de84b0ce7490ce5470be4450ae2430ae14109df3f08dd3d08dc3b07da3907d83706d63506d43305d23105"
    "d02f05ce2d04cc2b04ca2a04c82803c52603c32503c12302be2102bc2002b91e02b71d02b41b01b21a01af1801ac1701"
    "a91601a71401a41301a112019e10019b0f01980e01950d01920b018e0a018b09028808028507028106027e05027a0403"
), dtype=np.uint8).reshape(256, 3)
AUTH_WINDOW_S, AUTH_MAX = 300, 6

MISSION_ENUMS = {
    "task": ["SEARCH_SURVIVORS", "MAP_AREA", "GOTO_WAYPOINT", "RETURN_HOME", "HALT", "UNSUPPORTED"],
    "on_person_found": ["REPORT_AND_CONTINUE", "REPORT_AND_RETURN", "STAY"],
    "on_link_loss": ["CONTINUE_THEN_RETURN", "RETURN_NOW", "CONTINUE", "HOLD"],
    "first_branch": ["ANY", "LEFT", "RIGHT", "STRAIGHT"],
    "speed": ["NORMAL", "SLOW"],
}
MISSION_BOOLS = ["report_person", "approach_person", "return_when_done", "understood"]
MISSION_NUMS = {"max_duration_min": (1, 240), "max_distance_m": (5, 2000), "min_battery_pct": (5, 90)}


def log(*a):
    sys.stdout.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------ helpers
def png_palette(idx, palette, alpha):
    """Palette PNG from a uint8 index image (row 0 = top). Pure zlib, no PIL needed."""
    h, w = idx.shape
    raw = b"".join(b"\x00" + idx[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
            + chunk(b"PLTE", b"".join(bytes(c) for c in palette))
            + chunk(b"tRNS", bytes(alpha))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    """2D pose composition a*b, poses as (x, y, yaw)."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (a[0] + ca * b[0] - sa * b[1], a[1] + sa * b[0] + ca * b[1], a[2] + b[2])


def clean_frame(f):
    return (f or "").lstrip("/")


def sanitize_mission(m):
    """Whitelist every field. Anything unexpected is dropped or nulled, never passed through."""
    if not isinstance(m, dict):
        return None
    out = {}
    for k, allowed in MISSION_ENUMS.items():
        v = m.get(k)
        # an unknown task must never become a real one: it is UNSUPPORTED and gets refused
        out[k] = v if v in allowed else ("UNSUPPORTED" if k == "task" else allowed[0])
    for k in MISSION_BOOLS:
        out[k] = bool(m.get(k, False))
    for k, (lo, hi) in MISSION_NUMS.items():
        v = m.get(k)
        try:
            v = float(v)
            out[k] = v if lo <= v <= hi else None
        except (TypeError, ValueError):
            out[k] = None
    wp = m.get("waypoint")
    out["waypoint"] = wp if isinstance(wp, str) and len(wp) == 1 and wp.isalpha() else None
    out["text"] = str(m.get("text", ""))[:300]
    try:
        out["confidence"] = round(float(m.get("confidence", 0.0)), 3)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out


# ------------------------------------------------------------------------------------ parser
class StubParser(object):
    """Keyword fallback used only until the trained parser is installed in ~/dhwani_parser."""
    name = "keyword-fallback"

    def parse(self, text):
        t = " " + text.lower() + " "
        m = {"task": "UNSUPPORTED", "report_person": True, "approach_person": True,
             "on_person_found": "REPORT_AND_CONTINUE", "on_link_loss": "CONTINUE_THEN_RETURN",
             "return_when_done": True, "first_branch": "ANY", "speed": "NORMAL", "waypoint": None,
             "max_duration_min": None, "max_distance_m": None, "min_battery_pct": None,
             "confidence": 0.5, "understood": False, "text": text}
        import re
        wp = re.search(r"\b(?:go|drive|move|head|navigate)\s+(?:back\s+)?to\s+(?:the\s+)?(?:point|waypoint|spot)?\s*([a-z])\b", t)
        if wp and not any(w in t for w in (" stop ", " halt ")):
            m.update(task="GOTO_WAYPOINT", waypoint=wp.group(1).upper(), on_link_loss="RETURN_NOW",
                     return_when_done=False, understood=True)
            return m
        if any(w in t for w in (" stop ", " halt ", " freeze ")):
            m.update(task="HALT", return_when_done=False)
        elif any(w in t for w in ("person", "people", "survivor", "alive", "anyone", "someone", "victim")):
            m.update(task="SEARCH_SURVIVORS")
        elif any(w in t for w in (" come back", " return", " home ")):
            m.update(task="RETURN_HOME", return_when_done=False)
        elif any(w in t for w in (" map ", "explore", "survey", "scan the")):
            m.update(task="MAP_AREA", approach_person=False)
        m["understood"] = m["task"] != "UNSUPPORTED"
        return m


class ParserHolder(object):
    """Loads the trained TinyBERT parser if it exists, re-checks every minute (so the model can be
    installed tonight without restarting the gateway), and falls back to keywords."""

    def __init__(self, path):
        self.path = path
        self.impl = StubParser()
        self.kind = StubParser.name
        self.last_try = 0.0
        self.lock = threading.Lock()
        self.try_load()

    def try_load(self):
        self.last_try = time.time()
        mdir = os.path.join(self.path, "models", "tinybert4")
        if not os.path.exists(os.path.join(self.path, "dhwani_parser.py")) or not os.path.isdir(mdir):
            return
        try:
            if self.path not in sys.path:
                sys.path.insert(0, self.path)
            import importlib
            mod = importlib.import_module("dhwani_parser")
            impl = mod.MissionParser(mdir)
            impl.parse("stop")
            with self.lock:
                self.impl, self.kind = impl, "tinybert4"
            log("parser: trained TinyBERT parser loaded from", mdir)
        except Exception as e:
            log("parser: trained parser present but failed to load:", repr(e))

    def parse(self, text):
        if self.kind == StubParser.name and time.time() - self.last_try > 60:
            self.try_load()
        t0 = time.time()
        with self.lock:
            m = self.impl.parse(text)
        m["parse_ms"] = round((time.time() - t0) * 1000.0, 1)
        m["parser"] = self.kind
        return m


# ------------------------------------------------------------------------------------ ROS side
class Rover(object):
    """Everything the dashboard shows, kept current by ROS callbacks (their own threads)."""

    def __init__(self, gw, dry):
        self.gw = gw
        self.dry = dry
        self.lock = threading.Lock()
        self.tf = {}                      # (parent, child) -> (x, y, yaw)
        self.map_msg = None               # latest encoded map dict
        self.scan = None
        self.odom_v = (0.0, 0.0)
        self.mission_state = "?"
        self.exec_state = None
        self.exec_t = 0.0
        self.oak_status = None
        self.events = collections.deque(maxlen=60)
        self.junctions = []
        self.plan = []
        self.plan_t = 0.0
        self.trail = collections.deque(maxlen=600)
        self.counts = collections.defaultdict(int)
        self.rates = {}
        self._rate_t = time.time()
        self.cam_frame = None
        self.depth_sub = None
        self.guard_ahead, self.guard_t = float("inf"), 0.0
        self.guard_close_t, self.guard_close_d = 0.0, float("inf")
        self.guard_stop = GUARD_STOP

        def T(name):
            return name + "_dryrun" if dry else name
        self.pub_cmd = rospy.Publisher("/dashboard/cmd_vel_dryrun" if dry else "/cmd_vel", Twist, queue_size=1)
        self.pub_mcmd = rospy.Publisher(T("/mission/command"), String, queue_size=5)
        self.pub_permit = rospy.Publisher(T("/mission/permit"), Bool, queue_size=2)
        self.pub_plan = rospy.Publisher(T("/mission/plan"), String, queue_size=2, latch=True)
        self.pub_arm = rospy.Publisher(T("/executive/arm"), Bool, queue_size=2, latch=True)
        # In --dry-run the executive may still receive plan/arm on the REAL topics, but only while it
        # reports itself in TEST MODE (live false): then it decides and reports but sends no goals, so
        # the whole instruction -> brain -> decision chain can be shown on a parked rover.
        self.pub_plan_real = rospy.Publisher("/mission/plan", String, queue_size=2, latch=True) if dry else self.pub_plan
        self.pub_arm_real = rospy.Publisher("/executive/arm", Bool, queue_size=2, latch=True) if dry else self.pub_arm
        self.pub_cancel = rospy.Publisher(T("/move_base/cancel"), GoalID, queue_size=2)
        self.pub_hb = rospy.Publisher("/dashboard/heartbeat", String, queue_size=2)
        self.pub_batt = rospy.Publisher(T("/executive/battery"), Float32, queue_size=2)

        rospy.Subscriber("/tf", TFMessage, self.on_tf, queue_size=50)
        rospy.Subscriber("/tf_static", TFMessage, self.on_tf, queue_size=10)
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber("/scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/mission/state", String, self.on_mstate, queue_size=5)
        rospy.Subscriber("/events", String, self.on_event, queue_size=20)
        rospy.Subscriber("/junctions/poses", PoseArray, self.on_junctions, queue_size=1)
        rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.on_plan, queue_size=1)
        rospy.Subscriber("/executive/state", String, self.on_exec, queue_size=5)
        rospy.Subscriber("/oak/status", String, self.on_oak, queue_size=2)
        rospy.Subscriber("/oak/scan", LaserScan, self.on_oak_scan, queue_size=1)
        self.cam_sub = None

    def exec_test_mode(self):
        return bool(self.exec_state) and self.exec_state.get("live") is False and time.time() - self.exec_t < 5.0

    def publish_plan(self, s):
        (self.pub_plan_real if self.exec_test_mode() else self.pub_plan).publish(String(s))

    def publish_arm(self, on):
        (self.pub_arm_real if self.exec_test_mode() else self.pub_arm).publish(Bool(on))
        if self.dry and on is False:
            self.pub_arm_real.publish(Bool(False))        # disarming is always safe on the real topic

    # -- callbacks --
    def on_tf(self, msg):
        with self.lock:
            for t in msg.transforms:
                tr = t.transform
                self.tf[(clean_frame(t.header.frame_id), clean_frame(t.child_frame_id))] = (
                    tr.translation.x, tr.translation.y, quat_yaw(tr.rotation))

    def lookup(self, chain):
        with self.lock:
            p = (0.0, 0.0, 0.0)
            for a, b in zip(chain, chain[1:]):
                t = self.tf.get((a, b))
                if t is None:
                    return None
                p = compose(p, t)
            return p

    def robot_pose(self):
        return self.lookup(["map", "odom", "base_footprint"])

    def on_map(self, m):
        try:
            w, h = m.info.width, m.info.height
            g = np.asarray(m.data, dtype=np.int8).reshape(h, w)
            known = np.argwhere(g >= 0)
            if known.size == 0:
                return
            pad = 20
            r0, c0 = np.maximum(known.min(0) - pad, 0)
            r1, c1 = np.minimum(known.max(0) + pad + 1, [h, w])
            sub = g[r0:r1, c0:c1]
            idx = np.zeros(sub.shape, np.uint8)
            idx[(sub >= 0) & (sub < 25)] = 1
            idx[(sub >= 25) & (sub < 65)] = 2
            idx[sub >= 65] = 3
            png = png_palette(np.ascontiguousarray(idx[::-1]),
                              [(0, 0, 0), (214, 219, 226), (140, 148, 160), (24, 28, 36)],
                              [0, 255, 255, 255])
            res = m.info.resolution
            d = {"t": "map", "png": base64.b64encode(png).decode("ascii"), "res": res,
                 "ox": m.info.origin.position.x + c0 * res, "oy": m.info.origin.position.y + r0 * res,
                 "w": int(c1 - c0), "h": int(r1 - r0), "stamp": time.time()}
            with self.lock:
                self.map_msg = d
            self.gw.push_threadsafe(d)
        except Exception as e:
            log("map encode failed:", repr(e))

    def on_scan(self, s):
        self.counts["scan"] += 1
        with self.lock:
            self.scan = s

    def on_odom(self, o):
        self.counts["odom"] += 1
        self.odom_v = (o.twist.twist.linear.x, o.twist.twist.angular.z)

    def on_mstate(self, m):
        self.mission_state = m.data
        self.gw.push_threadsafe({"t": "mission_state", "state": m.data})

    def on_event(self, m):
        try:
            e = json.loads(m.data)
        except ValueError:
            e = {"text": m.data}
        e["rx"] = time.time()
        with self.lock:
            self.events.append(e)
        self.gw.push_threadsafe({"t": "event", "event": e})

    def on_junctions(self, pa):
        pts = [[round(p.position.x, 2), round(p.position.y, 2)] for p in pa.poses]
        with self.lock:
            self.junctions = pts
        self.gw.push_threadsafe({"t": "junctions", "pts": pts})

    def on_plan(self, p):
        now = time.time()
        if now - self.plan_t < 0.5:
            return
        self.plan_t = now
        poses = p.poses[::max(1, len(p.poses) // 150)]
        pts = [[round(q.pose.position.x, 2), round(q.pose.position.y, 2)] for q in poses]
        with self.lock:
            self.plan = pts
        self.gw.push_threadsafe({"t": "plan", "pts": pts})

    def on_exec(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        self.exec_state, self.exec_t = d, time.time()
        self.gw.push_threadsafe({"t": "exec", "state": d})

    def on_oak(self, m):
        self.oak_status = m.data
        self.gw.push_threadsafe({"t": "oak", "status": m.data})

    # -- camera (CSI via a C++ throttle node, started only while someone is watching) --
    def camera_wanted(self, on):
        if on and self.cam_sub is None:
            self.cam_sub = rospy.Subscriber("/dashboard/csi_throttled", Image, self.on_image,
                                            queue_size=1, buff_size=2 ** 22)
            self.gw.start_throttle()
        elif not on and self.cam_sub is not None:
            self.cam_sub.unregister()
            self.cam_sub = None
            self.gw.stop_throttle()

    def on_image(self, im):
        try:
            from PIL import Image as PILImage
            if im.encoding not in ("rgb8", "bgr8"):
                return
            img = PILImage.frombytes("RGB", (im.width, im.height), bytes(im.data))
            if im.encoding == "bgr8":
                b, g, r = img.split()
                img = PILImage.merge("RGB", (r, g, b))
            img = img.resize((CAM_W, CAM_H), PILImage.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=60)
            self.gw.camera_threadsafe(0, buf.getvalue())
        except Exception as e:
            log("camera encode failed:", repr(e))

    # -- drop guard for manual driving: the depth camera's drop and obstacle scan --
    def on_oak_scan(self, s):
        best, a = float("inf"), s.angle_min
        for r in s.ranges:
            if r == r and 0.0 < r < float("inf"):       # skip NaN and "nothing seen"
                fwd, lat = r * math.cos(a), r * math.sin(a)
                if fwd > 0 and abs(lat) <= GUARD_CORRIDOR:
                    best = min(best, fwd)
            a += s.angle_increment
        self.guard_ahead, self.guard_t = best, time.time()
        if best <= self.guard_stop:
            self.guard_close_t, self.guard_close_d = self.guard_t, best

    def guard_block(self):
        """Why forward driving is refused right now, or None."""
        if not (self.oak_status or "").startswith("OK") or time.time() - self.guard_t > GUARD_STALE_S:
            return "Forward blocked: the depth camera is not streaming, so drops cannot be seen."
        if time.time() - self.guard_close_t <= GUARD_HOLD_S:
            return "Forward blocked: drop or obstacle %.2f m ahead. Reverse or steer away." % self.guard_close_d
        return None

    # -- depth (OAK-D, from oak_node.py, subscribed only while someone is watching) --
    def depth_wanted(self, on):
        if on and self.depth_sub is None:
            self.depth_sub = rospy.Subscriber("/oak/depth/image_raw", Image, self.on_depth,
                                              queue_size=1, buff_size=2 ** 20)
            log("depth: subscribed")
        elif not on and self.depth_sub is not None:
            self.depth_sub.unregister()
            self.depth_sub = None
            log("depth: unsubscribed")

    def on_depth(self, im):
        try:
            from PIL import Image as PILImage
            if im.encoding != "16UC1":
                return
            mm = np.frombuffer(bytes(im.data), dtype="<u2").reshape(im.height, im.width)
            t = (mm.astype(np.float32) - DEPTH_NEAR_MM) * (255.0 / (DEPTH_FAR_MM - DEPTH_NEAR_MM))
            rgb = DEPTH_LUT[np.clip(t, 0, 255).astype(np.uint8)]
            rgb[mm == 0] = 0                                   # no reading: black
            buf = io.BytesIO()
            PILImage.fromarray(rgb, "RGB").save(buf, "JPEG", quality=80)
            self.gw.depth_threadsafe(buf.getvalue())
        except Exception as e:
            log("depth encode failed:", repr(e))

    # -- derived data for the periodic pushes --
    def pose_msg(self):
        p = self.robot_pose()
        if p is None:
            return None
        v, w = self.odom_v
        if not self.trail or math.hypot(p[0] - self.trail[-1][0], p[1] - self.trail[-1][1]) > 0.1:
            self.trail.append((round(p[0], 2), round(p[1], 2)))
        return {"t": "pose", "x": round(p[0], 3), "y": round(p[1], 3), "yaw": round(p[2], 3),
                "v": round(v, 3), "w": round(w, 3)}

    def scan_msg(self):
        with self.lock:
            s = self.scan
        if s is None:
            return None
        base = self.robot_pose()
        laser = None
        with self.lock:
            laser = self.tf.get(("base_footprint", clean_frame(s.header.frame_id)))
        if base is None or laser is None:
            return None
        p = compose(base, laser)
        step = 5
        r = np.asarray(s.ranges[::step], dtype=np.float32)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment * step
        ok = np.isfinite(r) & (r > s.range_min) & (r < s.range_max)
        r, a = r[ok], a[ok]
        xs = p[0] + r * np.cos(a + p[2])
        ys = p[1] + r * np.sin(a + p[2])
        pts = np.round(np.stack([xs, ys], 1), 2).tolist()
        return {"t": "scan", "pts": pts}

    def stats_msg(self):
        now = time.time()
        dt = max(0.2, now - getattr(self, "_rate_t", now - 1.0))
        self._rate_t = now
        for k in ("scan", "odom"):
            self.rates[k] = round(self.counts[k] / dt, 1)
            self.counts[k] = 0
        return {"t": "stats", "scan_hz": self.rates.get("scan", 0), "odom_hz": self.rates.get("odom", 0),
                "cpu": SysStats.cpu(), "mem": SysStats.mem(), "temp": SysStats.temp(),
                "load": SysStats.load(), "exec_alive": (now - self.exec_t) < 5.0,
                "mission_state": self.mission_state}


class SysStats(object):
    _prev = None

    @classmethod
    def cpu(cls):
        try:
            with open("/proc/stat") as f:
                parts = [float(x) for x in f.readline().split()[1:]]
            idle, total = parts[3] + parts[4], sum(parts)
            prev, cls._prev = cls._prev, (idle, total)
            if prev is None or total == prev[1]:
                return None
            return round(100.0 * (1.0 - (idle - prev[0]) / (total - prev[1])), 1)
        except Exception:
            return None

    @staticmethod
    def mem():
        try:
            d = {}
            with open("/proc/meminfo") as f:
                for ln in f:
                    k, v = ln.split(":")
                    d[k] = float(v.split()[0])
            return {"used_mb": round((d["MemTotal"] - d["MemAvailable"]) / 1024.0),
                    "total_mb": round(d["MemTotal"] / 1024.0)}
        except Exception:
            return None

    @staticmethod
    def temp():
        best = None
        for z in range(10):
            try:
                with open("/sys/devices/virtual/thermal/thermal_zone%d/type" % z) as f:
                    name = f.read().strip()
                if name != "CPU-therm":
                    continue
                with open("/sys/devices/virtual/thermal/thermal_zone%d/temp" % z) as f:
                    best = round(int(f.read()) / 1000.0, 1)
            except Exception:
                continue
        return best

    @staticmethod
    def load():
        try:
            with open("/proc/loadavg") as f:
                return float(f.read().split()[0])
        except Exception:
            return None


# ------------------------------------------------------------------------------------ gateway
class Client(object):
    def __init__(self, ws, ip):
        self.ws = ws
        self.ip = ip
        self.id = secrets.token_hex(4)
        self.token = None
        self.name = None
        self.camera = False
        self.depth = False
        self.rtt_ms = None


class Gateway(object):
    def __init__(self, args):
        self.args = args
        self.dry = args.dry_run
        self.loop = asyncio.get_event_loop()
        self.clients = set()
        self.tokens = {}                 # token -> {"name", "ip", "t"}
        self.controller = None           # {"token", "name", "t", "client_id"}
        self.auth_log = collections.defaultdict(list)
        self.drive = (0.0, 0.0, 0.0)     # v, w, time
        self.guard_note_t = 0.0
        self.driving = False
        self.halted = False              # after STOP, permission is withheld until "resume"
        self.pin = self.load_pin()
        self.parser = ParserHolder(os.path.expanduser(args.parser_dir))
        self.throttle_proc = None
        self.cam_last = {0: 0.0, 1: 0.0}
        self.objects = (0.0, [])         # (receive time, last OAK detections) for the Vision overlay
        self.audit = open(os.path.join(HERE, "logs", "commands.log"), "a")
        self.rover = Rover(self, self.dry)
        self.rover.guard_stop = max(GUARD_STOP, args.guard_stop)   # never closer than the default
        SysStats.cpu()

    # -- persistence --
    def load_pin(self):
        p = os.path.join(HERE, "pin.txt")
        pin = os.environ.get("DHWANI_PIN")
        if not pin and os.path.exists(p):
            pin = open(p).read().strip()
        if not pin:
            pin = "%06d" % random.SystemRandom().randint(0, 999999)
            with open(p, "w") as f:
                f.write(pin + "\n")
            os.chmod(p, 0o600)
        return pin

    def public_url(self):
        try:
            return open(os.path.join(HERE, "current_link.txt")).read().strip()
        except Exception:
            return None

    def note(self, client, what, **kw):
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": client.ip if client else "-",
               "name": client.name if client else "-", "cmd": what, "dry_run": self.dry}
        rec.update(kw)
        self.audit.write(json.dumps(rec) + "\n")
        self.audit.flush()

    # -- thread-safe entry points for ROS callbacks --
    def push_threadsafe(self, d):
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.broadcast(d)))

    def camera_threadsafe(self, src, jpeg):
        self.cam_last[src] = time.time()
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.send_camera(src, jpeg)))

    def depth_threadsafe(self, jpeg):
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.send_camera(2, jpeg, depth=True)))

    # -- throttle node for the CSI camera --
    def start_throttle(self):
        if self.throttle_proc is None or self.throttle_proc.poll() is not None:
            self.throttle_proc = subprocess.Popen(
                ["rosrun", "topic_tools", "throttle", "messages", "/csi_cam_0/image_raw",
                 str(CAM_FPS), "/dashboard/csi_throttled", "__name:=dashboard_cam_throttle"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("camera: throttle started")

    def stop_throttle(self):
        if self.throttle_proc is not None and self.throttle_proc.poll() is None:
            self.throttle_proc.terminate()
            log("camera: throttle stopped")
        self.throttle_proc = None

    # -- sending --
    # 28 Sep 2026: one viewer whose tab went to sleep (a phone on a hotspot) used to hold up every
    # broadcast for up to 30 s, because each send waited for that viewer's socket to drain in turn.
    # That froze map, stats and camera for everyone and queued the next command of whoever caused
    # the broadcast (a parse after Take control "worked once, then never"). Now every viewer is
    # sent to in parallel, no send waits longer than SEND_WAIT_S, and a viewer that is already
    # SLOW_BUF behind skips the frequent updates (pose, scan, map, stats) until it catches up.
    async def _send_one(self, c, s, critical):
        try:
            buf = c.ws.transport.get_write_buffer_size() if c.ws.transport else 0
        except Exception:
            buf = 0
        busy = getattr(c, "inflight", 0)
        if not critical and (buf > SLOW_BUF or busy >= 2):
            return                    # already behind: skip this frequent update
        c.inflight = busy + 1
        try:
            await asyncio.wait_for(c.ws.send(s), SEND_WAIT_S)
        except Exception:
            pass
        finally:
            c.inflight -= 1

    async def send(self, c, d):
        await self._send_one(c, json.dumps(d, separators=(",", ":")), True)

    async def broadcast(self, d):
        if self.clients:
            s = json.dumps(d, separators=(",", ":"))
            critical = d.get("t") in CRITICAL_MSGS
            await asyncio.gather(*[self._send_one(c, s, critical) for c in list(self.clients)])

    async def send_camera(self, src, jpeg, depth=False):
        frame = b"J" + bytes([src]) + jpeg     # src 0 CSI camera, 1 OAK colour, 2 OAK depth
        # a slow viewer drops frames instead of lagging, and never holds up the others
        await asyncio.gather(*[self._send_one(c, frame, False) for c in list(self.clients)
                               if (c.depth if depth else c.camera)])

    async def send_objects(self, objs, t):
        """OAK detections for the camera overlay, to viewers who have the camera on."""
        s = json.dumps({"t": "dets", "ts": t, "objects": objs[:40]}, separators=(",", ":"))
        await asyncio.gather(*[self._send_one(c, s, False) for c in list(self.clients) if c.camera])

    # -- control plane --
    def is_controller(self, c, token):
        ctl = self.controller
        return bool(token and ctl and ctl["token"] == token and token in self.tokens)

    def controller_state(self):
        ctl = self.controller
        if ctl and time.time() - ctl["t"] > CONTROL_STALE_S:
            self.controller = ctl = None
        return {"t": "control", "controller": ctl["name"] if ctl else None,
                "controller_client": ctl["client_id"] if ctl else None,
                "halted": self.halted, "dry_run": self.dry}

    def do_stop(self, client, why):
        """Open to everyone. Cancel goals, disarm the executive, tell mission_node, zero velocity."""
        self.halted = True
        self.driving = False
        self.drive = (0.0, 0.0, 0.0)
        r = self.rover
        r.pub_cancel.publish(GoalID())
        r.publish_arm(False)
        r.pub_mcmd.publish(String("STOP"))
        for _ in range(5):
            r.pub_cmd.publish(Twist())
        self.note(client, "STOP", why=why)
        log("STOP from", client.name if client else "?", client.ip if client else "", "-", why)

    async def handle(self, c, d):
        t = d.get("t")
        tok = d.get("token")
        if c.token and self.controller and self.controller["token"] == c.token:
            self.controller["t"] = time.time()
        if t == "ping":
            await self.send(c, {"t": "pong", "ts": d.get("ts")})
            if "rtt" in d:
                c.rtt_ms = d.get("rtt")
            return
        if t == "stop":
            self.do_stop(c, d.get("why", "STOP pressed"))
            await self.broadcast({"t": "toast", "level": "stop",
                                  "msg": "STOP by %s" % (c.name or "a viewer")})
            await self.broadcast(self.controller_state())
            return
        if t == "camera":
            c.camera = bool(d.get("on"))
            want_csi = any(x.camera and time.time() - self.cam_last[1] > 3.0 for x in self.clients)
            self.rover.camera_wanted(want_csi)
            return
        if t == "depth":
            c.depth = bool(d.get("on"))
            self.rover.depth_wanted(any(x.depth for x in self.clients))
            return
        if t == "auth":
            now = time.time()
            hist = [x for x in self.auth_log[c.ip] if now - x < AUTH_WINDOW_S]
            self.auth_log[c.ip] = hist
            if len(hist) >= AUTH_MAX:
                await self.send(c, {"t": "auth", "ok": False, "msg": "Too many wrong PINs. Wait 5 minutes."})
                return
            name = str(d.get("name") or "operator")[:24]
            if not secrets.compare_digest(str(d.get("pin", "")), self.pin):
                hist.append(now)            # only wrong PINs count toward the lockout
            if secrets.compare_digest(str(d.get("pin", "")), self.pin):
                token = secrets.token_hex(16)
                self.tokens[token] = {"name": name, "ip": c.ip, "t": now}
                c.token, c.name = token, name
                self.note(c, "AUTH_OK")
                await self.send(c, {"t": "auth", "ok": True, "token": token, "name": name})
            else:
                self.note(c, "AUTH_FAIL")
                await self.send(c, {"t": "auth", "ok": False, "msg": "Wrong PIN."})
            return
        if t == "parse":
            text = str(d.get("text", ""))[:300]
            m = await self.loop.run_in_executor(None, self.parser.parse, text)
            self.note(c, "PARSE", text=text, task=m.get("task"))
            await self.send(c, {"t": "parsed", "mission": m, "req": d.get("req")})
            return

        # everything below needs the PIN
        if not tok or tok not in self.tokens:
            await self.send(c, {"t": "ack", "cmd": t, "ok": False, "msg": "Enter the PIN to control."})
            return
        info = self.tokens[tok]
        c.token, c.name = tok, info["name"]

        if t == "take_control":
            self.controller_state()
            if self.controller and self.controller["token"] != tok and not d.get("force"):
                await self.send(c, {"t": "ack", "cmd": t, "ok": False, "busy": True,
                                    "msg": "%s is driving. Take over?" % self.controller["name"]})
                return
            self.controller = {"token": tok, "name": info["name"], "t": time.time(), "client_id": c.id}
            self.note(c, "TAKE_CONTROL", force=bool(d.get("force")))
            await self.broadcast(self.controller_state())
            return
        if t == "release":
            if self.controller and self.controller["token"] == tok:
                self.controller = None
                self.drive = (0.0, 0.0, 0.0)
            await self.broadcast(self.controller_state())
            return
        if not self.is_controller(c, tok):
            await self.send(c, {"t": "ack", "cmd": t, "ok": False, "msg": "Press Take control first."})
            return
        self.controller["t"] = time.time()

        if t == "drive":
            busy = self.rover.mission_state.startswith("DRIVING")
            ex = self.rover.exec_state or {}
            if busy or (ex.get("armed") and ex.get("phase") not in (None, "IDLE", "DONE", "HOLD")):
                await self.send(c, {"t": "ack", "cmd": t, "ok": False,
                                    "msg": "A mission is running. Press STOP first."})
                return
            if self.halted:
                await self.send(c, {"t": "ack", "cmd": t, "ok": False, "halted": True,
                                    "msg": "Stopped. Press Resume to drive again."})
                return
            v = max(-MAX_V, min(MAX_V, float(d.get("v", 0.0)) * MAX_V))
            w = max(-MAX_W, min(MAX_W, float(d.get("w", 0.0)) * MAX_W))
            self.drive = (v, w, time.time())
            return
        if t == "resume":
            self.halted = False
            self.note(c, "RESUME")
            await self.broadcast(self.controller_state())
            return
        if t == "start_mission":
            m = sanitize_mission(d.get("mission"))
            if not m or m["task"] == "UNSUPPORTED" or not m["understood"]:
                await self.send(c, {"t": "ack", "cmd": t, "ok": False, "msg": "That mission was not understood."})
                return
            await self.start_mission(c, m)
            return
        if t == "home":
            await self.start_mission(c, sanitize_mission({"task": "RETURN_HOME", "understood": True,
                                                          "text": "return home (button)"}))
            return
        if t == "battery":
            try:
                v = float(d.get("pct"))
            except (TypeError, ValueError):
                v = -1
            if not 5 <= v <= 100:
                await self.send(c, {"t": "ack", "cmd": t, "ok": False, "msg": "Battery must be 5 to 100 %."})
                return
            self.rover.pub_batt.publish(Float32(v))
            self.note(c, "BATTERY", pct=v)
            await self.send(c, {"t": "ack", "cmd": t, "ok": True, "msg": "Battery set to %.0f %%." % v})
            return
        if t == "arm":
            on = bool(d.get("on"))
            self.rover.publish_arm(on)
            if on:
                self.halted = False
            self.note(c, "ARM" if on else "DISARM")
            await self.broadcast({"t": "toast", "level": "info",
                                  "msg": ("Executive ARMED by %s" if on else "Executive disarmed by %s") % c.name})
            return

    async def start_mission(self, c, m):
        """Hand the mission to the executive if it is alive; otherwise use mission_node for the
        tasks it can do by itself (HALT, HOME, GOTO <waypoint>)."""
        r = self.rover
        exec_alive = (time.time() - r.exec_t) < 5.0
        self.halted = False
        if m["task"] == "HALT":
            self.do_stop(c, "mission HALT")
            await self.send(c, {"t": "ack", "cmd": "start_mission", "ok": True, "msg": "Stopped."})
            return
        if exec_alive:
            r.publish_plan(json.dumps(m))
            self.note(c, "PLAN", mission=m)
            await self.broadcast({"t": "toast", "level": "info",
                                  "msg": "Mission sent by %s: %s" % (c.name, m["task"])})
            await self.send(c, {"t": "ack", "cmd": "start_mission", "ok": True, "need_arm": True,
                                "msg": "Mission loaded. Press ARM to let the rover drive."})
            return
        if m["task"] == "RETURN_HOME":
            r.pub_mcmd.publish(String("HOME"))
            self.note(c, "HOME")
            await self.send(c, {"t": "ack", "cmd": "start_mission", "ok": True, "msg": "Going home (waypoint A)."})
            return
        if m["task"] == "GOTO_WAYPOINT" and m.get("waypoint"):
            r.pub_mcmd.publish(String("GOTO %s" % m["waypoint"]))
            self.note(c, "GOTO", wp=m["waypoint"])
            await self.send(c, {"t": "ack", "cmd": "start_mission", "ok": True,
                                "msg": "Going to waypoint %s." % m["waypoint"]})
            return
        await self.send(c, {"t": "ack", "cmd": "start_mission", "ok": False,
                            "msg": "The on-board executive is not running, so %s cannot start." % m["task"]})

    # -- periodic tasks --
    async def drive_loop(self):
        """20 Hz. Publishes the held command while it is fresh; one zero burst when it goes stale."""
        while True:
            await asyncio.sleep(0.05)
            v, w, t = self.drive
            fresh = (time.time() - t) < DRIVE_STALE_S and not self.halted and self.controller
            if fresh:
                why = self.rover.guard_block() if (v > 0 and not self.args.no_drop_guard) else None
                if why:
                    v = 0.0                                   # forward refused; steering still passes
                    now = time.time()
                    if now - self.guard_note_t > 1.5:
                        self.guard_note_t = now
                        asyncio.ensure_future(self.broadcast({"t": "toast", "level": "warn", "msg": why}))
                tw = Twist()
                tw.linear.x, tw.angular.z = v, w
                self.rover.pub_cmd.publish(tw)
                self.driving = True
            elif self.driving:
                for _ in range(3):
                    self.rover.pub_cmd.publish(Twist())
                self.driving = False

    async def slow_loop(self):
        n = 0
        while True:
            await asyncio.sleep(0.1)
            n += 1
            if not self.clients:
                continue
            pm = self.rover.pose_msg()
            if pm:
                await self.broadcast(pm)
            if n % 3 == 0:
                sm = self.rover.scan_msg()
                if sm:
                    await self.broadcast(sm)
            if n % 5 == 0:
                ctl = self.controller_state()
                if self.controller and not self.halted:
                    self.rover.pub_permit.publish(Bool(True))
                    rtt = None
                    for c in self.clients:
                        if c.id == self.controller.get("client_id"):
                            rtt = c.rtt_ms
                    self.rover.pub_hb.publish(String(json.dumps(
                        {"controller": self.controller["name"], "rtt_ms": rtt, "t": time.time()})))
            if n % 10 == 0:
                st = self.rover.stats_msg()
                st["clients"] = len(self.clients)
                st["parser"] = self.parser.kind
                st["oak_live"] = time.time() - self.cam_last[1] < 3.0
                await self.broadcast(st)
                await self.broadcast(self.controller_state())
            if n % 600 == 0:
                now = time.time()
                for k in [k for k, v in self.tokens.items() if now - v["t"] > TOKEN_TTL_S]:
                    del self.tokens[k]

    # -- OAK preview arrives by UDP from the camera process --
    class OakUdp(asyncio.DatagramProtocol):
        def __init__(self, gw):
            self.gw = gw

        def datagram_received(self, data, addr):
            if data[:2] == b"\xff\xd8":
                self.gw.cam_last[1] = time.time()
                asyncio.ensure_future(self.gw.send_camera(1, data))

    class ObjUdp(asyncio.DatagramProtocol):
        """Every detection oak_v3.py makes (port 5605), relayed as {"t": "dets"} for the overlay."""
        def __init__(self, gw):
            self.gw = gw

        def datagram_received(self, data, addr):
            try:
                d = json.loads(data.decode())
            except Exception:
                return
            objs = d.get("objects") or []
            self.gw.objects = (time.time(), objs)
            asyncio.ensure_future(self.gw.send_objects(objs, d.get("t")))

    # -- http + websocket --
    async def process_request(self, path, headers):
        if path == "/ws":
            return None
        path = path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path == "/api/info":
            body = json.dumps({"version": VERSION, "dry_run": self.dry, "public_url": self.public_url(),
                               "parser": self.parser.kind}).encode()
            return HTTPStatus.OK, [("Content-Type", "application/json"), ("Cache-Control", "no-store")], body
        if path == "/qr.svg":
            url = self.public_url() or "http://192.168.55.1:%d/" % self.args.port
            import segno
            buf = io.BytesIO()
            segno.make(url, error="m").save(buf, kind="svg", scale=8, border=2, dark="#0b0e13", light="#ffffff")
            return HTTPStatus.OK, [("Content-Type", "image/svg+xml"), ("Cache-Control", "no-store")], buf.getvalue()
        f = os.path.normpath(os.path.join(HERE, "web", path.lstrip("/")))
        if not f.startswith(os.path.join(HERE, "web")) or not os.path.isfile(f):
            return HTTPStatus.NOT_FOUND, [("Content-Type", "text/plain")], b"not found"
        ctype = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
                 ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png",
                 ".json": "application/json", ".webmanifest": "application/manifest+json"}.get(
            os.path.splitext(f)[1], "application/octet-stream")
        with open(f, "rb") as fh:
            body = fh.read()
        return HTTPStatus.OK, [("Content-Type", ctype), ("Cache-Control", "no-cache")], body

    async def ws_handler(self, ws, path):
        ip = ws.request_headers.get("CF-Connecting-IP") or (ws.remote_address or ["?"])[0]
        c = Client(ws, ip)
        self.clients.add(c)
        log("client +", c.id, ip, "(%d connected)" % len(self.clients))
        try:
            r = self.rover
            await self.send(c, {"t": "hello", "version": VERSION, "dry_run": self.dry, "you": c.id,
                                "limits": {"v": MAX_V, "w": MAX_W}, "parser": self.parser.kind,
                                "public_url": self.public_url()})
            if r.map_msg:
                await self.send(c, r.map_msg)
            await self.send(c, {"t": "trail", "pts": list(r.trail)})
            await self.send(c, {"t": "junctions", "pts": r.junctions})
            await self.send(c, {"t": "plan", "pts": r.plan})
            await self.send(c, {"t": "events", "items": list(r.events)})
            await self.send(c, {"t": "mission_state", "state": r.mission_state})
            if r.exec_state:
                await self.send(c, {"t": "exec", "state": r.exec_state})
            if r.oak_status:
                await self.send(c, {"t": "oak", "status": r.oak_status})
            await self.send(c, self.controller_state())
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                try:
                    await self.handle(c, d)
                except Exception as e:
                    log("handler error:", repr(e))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(c)
            if self.controller and self.controller.get("client_id") == c.id:
                self.drive = (0.0, 0.0, 0.0)      # the drive loop stops the rover on the next tick
                self.controller = None            # and /mission/permit stops with the lock
                self.note(c, "CONTROLLER_DISCONNECTED")
                asyncio.ensure_future(self.broadcast(self.controller_state()))
            if not any(x.camera for x in self.clients):
                self.rover.camera_wanted(False)
            if not any(x.depth for x in self.clients):
                self.rover.depth_wanted(False)
            log("client -", c.id, "(%d connected)" % len(self.clients))

    def run(self):
        start = websockets.serve(self.ws_handler, "0.0.0.0", self.args.port,
                                 process_request=self.process_request, compression=None,
                                 max_size=2 ** 20, ping_interval=10, ping_timeout=20)
        self.loop.run_until_complete(start)
        self.loop.run_until_complete(self.loop.create_datagram_endpoint(
            lambda: Gateway.OakUdp(self), local_addr=("127.0.0.1", 5601)))
        self.loop.run_until_complete(self.loop.create_datagram_endpoint(
            lambda: Gateway.ObjUdp(self), local_addr=("127.0.0.1", OBJ_PORT)))
        asyncio.ensure_future(self.drive_loop())
        asyncio.ensure_future(self.slow_loop())
        log("gateway up on :%d  dry_run=%s  parser=%s" % (self.args.port, self.dry, self.parser.kind))
        self.loop.add_signal_handler(signal.SIGTERM, self.loop.stop)
        self.loop.add_signal_handler(signal.SIGINT, self.loop.stop)
        try:
            self.loop.run_forever()
        finally:
            for _ in range(3):
                self.rover.pub_cmd.publish(Twist())
            self.stop_throttle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="send motion to *_dryrun topics")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--no-drop-guard", action="store_true",
                    help="allow forward driving without the depth camera's drop check")
    ap.add_argument("--guard-stop", type=float, default=GUARD_STOP,
                    help="refuse forward driving when a drop or obstacle is this close (m, from the centre)")
    ap.add_argument("--parser-dir", default="~/dhwani_parser")
    args = ap.parse_args(rospy.myargv()[1:])
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    rospy.init_node("dhwani_dashboard", anonymous=True, disable_signals=True)
    Gateway(args).run()


if __name__ == "__main__":
    main()
