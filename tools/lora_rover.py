#!/usr/bin/env python
"""LoRa dead-man's switch for the rover. Halts if the base station goes silent.

WHY: on 2026-09-23 the rover drove 24 m toward the hall, dropped off Wi-Fi (-31 dBm -> -58 dBm
and then nothing), and could not be stopped. It had a live move_base goal the whole time. The
only reason it came back is that it eventually wandered into range on its own. The watchdog
that should have caught that lived on the operator's laptop, at the far end of the link that
had just failed - which is the one place it cannot work.

This node puts the watchdog ON THE ROVER and carries it over a link that degrades instead of
disappearing. 1 W of LoRa reaches through building structure that kills 2.4 GHz outright.

THE RULE IS INVERTED ON PURPOSE. The rover does not wait to be told to stop; it requires
continuous permission to keep moving. Silence means halt. A stop command that has to arrive is
useless exactly when the radio has failed, which is when you need it.

Budget: air rate is pinned at 2400 bps on board revision 10408-V1.1 (it will not store a lower
value). One ~44 byte line per second is ~350 bps on air, leaving plenty of headroom for the
base's replies and for retries.
"""
import math
import time
import serial
import rospy
import tf
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from actionlib_msgs.msg import GoalID

PORT       = rospy.get_param("~port", "/dev/ttyTHS1")   # Jetson 40-pin header UART
BAUD       = rospy.get_param("~baud", 9600)             # E32 UART default, NOT the air rate
PERMIT_S   = rospy.get_param("~permit_timeout", 5.0)    # no permission for this long -> halt
TX_HZ      = rospy.get_param("~tx_hz", 1.0)


class LoraDeadman(object):
    def __init__(self):
        self.ser = serial.Serial(PORT, BAUD, timeout=0)
        self.tf = tf.TransformListener()
        self.cmd_pub    = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self.link_pub   = rospy.Publisher("/lora/status", String, queue_size=1, latch=True)
        self.last_permit = time.time()
        self.halted = False
        self.seq = 0
        self.rxbuf = b""

    def pose(self):
        try:
            (t, q) = self.tf.lookupTransform("map", "base_footprint", rospy.Time(0))
            return t[0], t[1], math.degrees(tf.transformations.euler_from_quaternion(q)[2])
        except Exception:
            return float('nan'), float('nan'), float('nan')

    def halt(self, why):
        if not self.halted:
            rospy.logerr("LoRa deadman: %s - HALTING", why)
            self.link_pub.publish(String("HALT: %s" % why))
            self.halted = True
        self.cancel_pub.publish(GoalID())
        self.cmd_pub.publish(Twist())

    def resume(self):
        if self.halted:
            rospy.logwarn("LoRa deadman: permission restored (operator must re-send the goal)")
            self.link_pub.publish(String("OK"))
            self.halted = False

    def read_lines(self):
        try:
            n = self.ser.in_waiting
            if n:
                self.rxbuf += self.ser.read(n)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "LoRa serial read failed: %s", e)
            return []
        out = []
        while b"\n" in self.rxbuf:
            line, self.rxbuf = self.rxbuf.split(b"\n", 1)
            out.append(line.strip().decode("ascii", "ignore"))
        if len(self.rxbuf) > 512:      # never let a garbled link grow the buffer forever
            self.rxbuf = b""
        return out

    def run(self):
        rospy.loginfo("LoRa deadman on %s @ %d, permit timeout %.1fs", PORT, BAUD, PERMIT_S)
        self.link_pub.publish(String("starting"))
        rate = rospy.Rate(10)
        last_tx = 0.0
        while not rospy.is_shutdown():
            for line in self.read_lines():
                # B,<seq>,<cmd>
                if line.startswith("B,"):
                    parts = line.split(",")
                    cmd = parts[2].strip().upper() if len(parts) > 2 else ""
                    if cmd == "STOP":
                        self.halt("operator sent STOP")
                        self.last_permit = 0          # stay halted until an explicit GO
                    elif cmd == "GO":
                        self.last_permit = time.time()
                        self.resume()

            if time.time() - self.last_permit > PERMIT_S:
                self.halt("no permission for %.1f s (link lost or base stopped)" % PERMIT_S)

            now = time.time()
            if now - last_tx >= 1.0 / TX_HZ:
                last_tx = now
                self.seq += 1
                x, y, yaw = self.pose()
                state = "HALT" if self.halted else "RUN"
                msg = "R,%d,%.2f,%.2f,%.0f,%s\n" % (self.seq, x, y, yaw, state)
                try:
                    self.ser.write(msg.encode("ascii"))
                except Exception as e:
                    rospy.logwarn_throttle(5.0, "LoRa serial write failed: %s", e)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("lora_deadman")
    node = None
    try:
        node = LoraDeadman()
        node.run()
    except Exception as e:
        rospy.logfatal("LoRa deadman died: %s", e)
        # the whole point of this node is that its death must stop the robot
        cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        cancel = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        rospy.sleep(0.5)
        cancel.publish(GoalID())
        for _ in range(20):
            cmd.publish(Twist()); rospy.sleep(0.05)
        raise
