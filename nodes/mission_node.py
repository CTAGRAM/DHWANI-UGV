#!/usr/bin/env python
"""The rover's own mission executor. Starts at boot, takes orders over LoRa or Wi-Fi, no SSH.

WHAT THIS REPLACES: until now every autonomous run began with a human opening an SSH session and
launching a script by hand. The navigation was autonomous; the *starting* of it was not, and
neither was stopping it - on 2026-09-23 the rover drove 24 m to the hall, dropped off Wi-Fi, and
could not be halted because the only thing that could halt it lived on the laptop at the far end
of the link that had just failed.

PERMISSION IS INVERTED, and this is the whole safety design. The rover does not wait to be told
to stop. While a mission runs it requires a continuous stream of permission, and silence means
halt. A STOP that has to arrive is worthless exactly when the radio is gone.

Permission is accepted from EITHER transport, newest wins:
    LoRa   "B,<seq>,GO"  on /dev/ttyTHS1   - control plane, survives to hundreds of metres
    Wi-Fi  /mission/permit (Bool)           - data plane, dies at about 25 m indoors
That is the "LoRa primary, Wi-Fi fallback" arrangement: losing Wi-Fi costs you the map view,
losing both stops the robot. Permission is only required while a mission is ACTIVE - an idle
rover is not a hazard and should not need babysitting.

EVENTS go three places at once so an operator sees them however they are connected: /events as
JSON for logging, /events/markers for Foxglove, and a compact line over LoRa for when Wi-Fi is
gone. A person sighting is just an event with a position, so the alert path is already built and
tested before the camera that produces sightings exists.
"""
import json
import math
import os
import time

import rospy
import tf
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PointStamped
from actionlib_msgs.msg import GoalID
from visualization_msgs.msg import Marker, MarkerArray

from goto import Nav

PORT        = rospy.get_param("~lora_port", "/dev/ttyTHS1") if False else "/dev/ttyTHS1"
WAYPOINTS   = os.path.expanduser("~/waypoints.json")
PERMIT_S    = 6.0      # no permission from either transport for this long -> abort + halt
LEG_TIMEOUT = 180
TELEM_HZ    = 1.0


class Events(object):
    """One sighting, three audiences: JSON log, Foxglove marker, LoRa line."""

    def __init__(self, lora):
        self.lora = lora
        self.pub = rospy.Publisher("/events", String, queue_size=10, latch=True)
        self.mk = rospy.Publisher("/events/markers", MarkerArray, queue_size=10, latch=True)
        self.log = []
        self.n = 0

    def fire(self, kind, text, xy=None, colour=(1.0, 0.2, 0.2)):
        self.n += 1
        e = {"n": self.n, "t": round(rospy.get_time(), 1), "kind": kind, "text": text}
        if xy:
            e["x"], e["y"] = round(xy[0], 2), round(xy[1], 2)
        self.log.append(e)
        self.pub.publish(String(json.dumps(e)))
        rospy.loginfo("EVENT %s: %s%s", kind, text,
                      (" at (%.2f, %.2f)" % xy) if xy else "")
        if xy:
            self._marker(self.n, kind, text, xy, colour)
        if self.lora:
            # keep it tiny: the air rate is pinned at 2400 bps on this board revision
            self.lora.send("E,%d,%s,%s" % (self.n, kind,
                                           ("%.1f,%.1f" % xy) if xy else "-,-"))

    def _marker(self, i, kind, text, xy, colour):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"; m.header.stamp = rospy.Time.now()
        m.ns = "events"; m.id = i
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = xy[0], xy[1], 0.25
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.35
        m.color.r, m.color.g, m.color.b = colour; m.color.a = 0.9
        arr.markers.append(m)
        t = Marker()
        t.header = m.header; t.ns = "event_labels"; t.id = i
        t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = xy[0], xy[1], 0.6
        t.pose.orientation.w = 1.0
        t.scale.z = 0.25
        t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
        t.text = "%s: %s" % (kind, text)
        arr.markers.append(t)
        self.mk.publish(arr)


class Lora(object):
    """Line-oriented link over the E32. Absent hardware must not stop the node booting."""

    def __init__(self, port):
        self.ser = None
        self.buf = b""
        try:
            import serial
            self.ser = serial.Serial(port, 9600, timeout=0)
            rospy.loginfo("LoRa up on %s", port)
        except Exception as e:
            rospy.logwarn("LoRa unavailable (%s) - Wi-Fi only until the radio is wired", e)

    def send(self, line):
        if not self.ser:
            return
        try:
            self.ser.write((line + "\n").encode("ascii"))
        except Exception as e:
            rospy.logwarn_throttle(10.0, "LoRa write failed: %s", e)

    def lines(self):
        if not self.ser:
            return []
        try:
            n = self.ser.in_waiting
            if n:
                self.buf += self.ser.read(n)
        except Exception:
            return []
        out = []
        while b"\n" in self.buf:
            ln, self.buf = self.buf.split(b"\n", 1)
            out.append(ln.strip().decode("ascii", "ignore"))
        if len(self.buf) > 512:
            self.buf = b""      # a garbled link must not grow the buffer without bound
        return out


class Mission(object):
    def __init__(self):
        self.nav = Nav()
        time.sleep(2.0)
        self.tf = tf.TransformListener()
        self.lora = Lora(PORT)
        self.ev = Events(self.lora)
        self.state = "IDLE"
        self.active = False
        self.abort = False
        self.last_permit = 0.0
        self.persons = []

        self.state_pub = rospy.Publisher("/mission/state", String, queue_size=1, latch=True)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)

        rospy.Subscriber("/mission/command", String, self.on_command, queue_size=5)
        rospy.Subscriber("/mission/permit", Bool, self.on_permit, queue_size=5)
        # the OAK node will publish one PointStamped per person, in the map frame.
        # subscribing now means the alert path is live and testable before the camera is.
        rospy.Subscriber("/oak/person", PointStamped, self.on_person, queue_size=10)

        self.set_state("IDLE")

    # ---- plumbing ---------------------------------------------------------------------------
    def set_state(self, s):
        self.state = s
        self.state_pub.publish(String(s))

    def on_permit(self, msg):
        if msg.data:
            self.last_permit = time.time()

    def on_person(self, msg):
        """A sighting arrives in base_footprint and must be pinned to the map.

        oak_node.py runs under python3.6 because depthai has no cp27 wheel, and `tf` is a C
        extension with no python3 build on Melodic - so the camera node cannot transform its own
        detections. It publishes in the robot frame and the conversion happens here, under 2.7,
        in the node that already owns the map.
        """
        try:
            self.tf.waitForTransform("map", msg.header.frame_id or "base_footprint",
                                     rospy.Time(0), rospy.Duration(0.5))
            m = msg
            m.header.stamp = rospy.Time(0)
            msg = self.tf.transformPoint("map", m)
        except Exception as e:
            rospy.logwarn_throttle(10.0, "sighting dropped, no map transform: %s", e)
            return
        p = (msg.point.x, msg.point.y)
        for q in self.persons:
            if math.hypot(p[0] - q[0], p[1] - q[1]) < 1.0:
                return              # same person seen again, not a new sighting
        self.persons.append(p)
        self.ev.fire("PERSON", "person #%d" % len(self.persons), p, colour=(1.0, 0.1, 0.1))

    def on_command(self, msg):
        self.handle(msg.data)

    def handle(self, line):
        parts = line.strip().split()
        if not parts:
            return
        cmd = parts[0].upper()
        if cmd == "STOP":
            self.abort = True
            self.halt("operator STOP")
        elif cmd in ("GOTO", "TOUR", "HOME", "EXPLORE"):
            if self.active:
                rospy.logwarn("busy (%s); ignoring %s", self.state, cmd)
                return
            self.pending = (cmd, parts[1:])
        elif cmd == "PING":
            self.lora.send("P,%s" % self.state)

    def halt(self, why):
        self.cancel_pub.publish(GoalID())
        t = time.time()
        while time.time() - t < 1.0:
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)
        self.set_state("HALTED")
        self.ev.fire("HALT", why)

    def permitted(self):
        return (time.time() - self.last_permit) <= PERMIT_S

    def waypoints(self):
        try:
            return json.load(open(WAYPOINTS))
        except Exception:
            return {}

    # ---- missions ---------------------------------------------------------------------------
    def drive_to(self, name, wp):
        gx, gy = wp["x"], wp["y"]
        gyaw = math.radians(wp["yaw_deg"])
        if not self.nav.send(gx, gy, gyaw):
            self.ev.fire("REFUSED", "goal %s not reachable" % name, (gx, gy), (1, 0.6, 0))
            return False
        t0 = time.time()
        while time.time() - t0 < LEG_TIMEOUT:
            if self.abort:
                return False
            if not self.permitted():
                self.abort = True
                self.halt("permission lost (LoRa and Wi-Fi both silent %.0fs)" % PERMIT_S)
                return False
            st = self.nav.status[-1].status if self.nav.status else None
            if st == 3:
                x, y, _ = self.nav.pose()
                self.ev.fire("ARRIVED", name, (x, y), (0.2, 1.0, 0.4))
                return True
            if st in (4, 5):
                self.ev.fire("FAILED", "move_base gave up on %s" % name, (gx, gy), (1, 0.5, 0))
                return False
            time.sleep(0.25)
        self.ev.fire("TIMEOUT", name, (gx, gy), (1, 0.5, 0))
        return False

    def run_mission(self, cmd, args):
        self.active = True
        self.abort = False
        self.last_permit = time.time()      # the command itself grants the first permission
        wps = self.waypoints()
        self.ev.fire("START", "%s %s" % (cmd, " ".join(args)))
        try:
            if cmd == "GOTO":
                names = args[:1]
            elif cmd == "TOUR":
                names = args
            elif cmd == "HOME":
                names = ["A"] if "A" in wps else []
            else:
                self.ev.fire("ERROR", "%s not implemented in this node" % cmd)
                names = []
            missing = [n for n in names if n not in wps]
            if missing:
                self.ev.fire("ERROR", "unknown waypoint %s" % ",".join(missing))
                names = []
            for n in names:
                self.set_state("DRIVING %s" % n)
                if not self.drive_to(n, wps[n]):
                    break
            self.ev.fire("DONE", "%s finished" % cmd)
        finally:
            self.nav.stop()
            self.active = False
            self.set_state("HALTED" if self.abort else "IDLE")

    # ---- main -------------------------------------------------------------------------------
    def spin(self):
        self.pending = None
        rate = rospy.Rate(10)
        last_tx = 0.0
        while not rospy.is_shutdown():
            for ln in self.lora.lines():
                if ln.startswith("B,"):
                    p = ln.split(",")
                    c = p[2].strip().upper() if len(p) > 2 else ""
                    if c == "GO":
                        self.last_permit = time.time()
                    elif c == "STOP":
                        self.abort = True
                        self.halt("LoRa STOP")
                elif ln.startswith("C,"):
                    self.handle(ln[2:].replace(",", " "))

            now = time.time()
            if now - last_tx >= 1.0 / TELEM_HZ:
                last_tx = now
                try:
                    x, y, yaw = self.nav.pose()
                except Exception:
                    x = y = yaw = float("nan")
                self.lora.send("R,%.1f,%.1f,%.0f,%s" % (x, y, math.degrees(yaw), self.state))

            if self.pending and not self.active:
                cmd, args = self.pending
                self.pending = None
                self.run_mission(cmd, args)
            rate.sleep()


if __name__ == "__main__":
    # goto.Nav calls rospy.init_node("dhwani_goto") in its own constructor, so this module must
    # NOT call it first - two init_node calls with different names is a hard error.
    m = None
    try:
        m = Mission()
        rospy.loginfo("mission node up - commands on /mission/command or LoRa 'C,<cmd>'")
        m.spin()
    except Exception as e:
        rospy.logfatal("mission node died: %s", e)
        cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        cancel = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        rospy.sleep(0.5)
        cancel.publish(GoalID())
        for _ in range(20):
            cmd.publish(Twist()); rospy.sleep(0.05)
        raise
