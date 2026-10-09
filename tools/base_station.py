#!/usr/bin/env python3
"""Operator console for the rover. Holds its permission to move, sends missions, shows alerts.

TALKS ROSBRIDGE, NOT ROS. rosbridge is already running on the rover at :9090 and speaks plain
JSON over a websocket, so this needs no ROS installation on the laptop - which matters, because
there is no ROS for macOS worth the trouble. The same console will later drive the LoRa link by
swapping the transport; the command vocabulary is identical, deliberately.

PERMISSION IS THE POINT. While this is running and not stopped, it publishes /mission/permit at
2 Hz. The rover halts on its own if that stream stops for 6 seconds - so closing this window,
losing Wi-Fi, or walking out of range all stop the robot. That is the inversion that was missing
on 2026-09-23, when the rover drove out of range with a live goal and nothing could reach it.

    s        STOP now and hold        g   resume permission
    1        GOTO A                   h   HOME
    t A B A  TOUR through waypoints   q   quit (rover halts within 6 s)
"""
import argparse
import json
import sys
import threading
import time

try:
    from websocket import create_connection          # pip install websocket-client
except ImportError:
    sys.exit("need websocket-client:  python3 -m pip install websocket-client")


class Base:
    def __init__(self, url):
        self.ws = create_connection(url, timeout=5)
        self.permit = True
        self.running = True
        self.state = "?"
        self.pose = None
        self.events = []
        self.lock = threading.Lock()

    def send(self, obj):
        with self.lock:
            self.ws.send(json.dumps(obj))

    def advertise(self, topic, typ):
        self.send({"op": "advertise", "topic": topic, "type": typ})

    def subscribe(self, topic, typ):
        self.send({"op": "subscribe", "topic": topic, "type": typ, "throttle_rate": 200})

    def publish(self, topic, msg):
        self.send({"op": "publish", "topic": topic, "msg": msg})

    # ---- threads ---------------------------------------------------------------------------
    def permit_loop(self):
        """2 Hz, well inside the rover's 6 s timeout so a dropped packet is not a halt."""
        while self.running:
            if self.permit:
                self.publish("/mission/permit", {"data": True})
            time.sleep(0.5)

    def rx_loop(self):
        while self.running:
            try:
                raw = self.ws.recv()
            except Exception:
                if self.running:
                    print("\n[link lost - rover will halt within 6 s]")
                break
            try:
                m = json.loads(raw)
            except Exception:
                continue
            if m.get("op") != "publish":
                continue
            t, msg = m.get("topic"), m.get("msg", {})
            if t == "/mission/state":
                self.state = msg.get("data", "?")
            elif t == "/events":
                try:
                    e = json.loads(msg.get("data", "{}"))
                except Exception:
                    continue
                self.events.append(e)
                loc = (" at (%.2f, %.2f)" % (e["x"], e["y"])) if "x" in e else ""
                bell = "\a" if e.get("kind") == "PERSON" else ""
                print("%s\n  >> %-8s %s%s" % (bell, e.get("kind"), e.get("text", ""), loc))
            elif t == "/amcl_pose":
                p = msg.get("pose", {}).get("pose", {}).get("position", {})
                self.pose = (p.get("x", 0.0), p.get("y", 0.0))

    def status_loop(self):
        while self.running:
            pos = ("(%6.2f,%6.2f)" % self.pose) if self.pose else "   (no pose)  "
            n = sum(1 for e in self.events if e.get("kind") == "PERSON")
            sys.stdout.write("\r[%s] %-14s %s  persons:%d  " %
                             ("GO  " if self.permit else "STOP", self.state, pos, n))
            sys.stdout.flush()
            time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.55.1")
    ap.add_argument("--port", type=int, default=9090)
    a = ap.parse_args()
    url = "ws://%s:%d" % (a.host, a.port)
    print("connecting to %s ..." % url)
    b = Base(url)

    b.advertise("/mission/permit", "std_msgs/Bool")
    b.advertise("/mission/command", "std_msgs/String")
    b.subscribe("/mission/state", "std_msgs/String")
    b.subscribe("/events", "std_msgs/String")
    b.subscribe("/amcl_pose", "geometry_msgs/PoseWithCovarianceStamped")

    for fn in (b.permit_loop, b.rx_loop, b.status_loop):
        threading.Thread(target=fn, daemon=True).start()

    print(__doc__.split("    s ")[-1])
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            k = line.strip()
            low = k.lower()
            if low == "q":
                break
            elif low == "s":
                b.permit = False
                b.publish("/mission/command", {"data": "STOP"})
                print("\n>>> STOP sent and permission withheld")
            elif low == "g":
                b.permit = True
                print("\n>>> permission resumed")
            elif low == "1":
                b.publish("/mission/command", {"data": "GOTO A"})
                print("\n>>> GOTO A")
            elif low == "h":
                b.publish("/mission/command", {"data": "HOME"})
                print("\n>>> HOME")
            elif low.startswith("t "):
                cmd = "TOUR " + k[2:].strip()
                b.publish("/mission/command", {"data": cmd})
                print("\n>>> %s" % cmd)
            elif k:
                b.publish("/mission/command", {"data": k.upper()})
                print("\n>>> %s" % k.upper())
    except KeyboardInterrupt:
        pass
    b.running = False
    try:
        b.ws.close()
    except Exception:
        pass
    print("\nconsole closed - rover halts within 6 s")


if __name__ == "__main__":
    main()
