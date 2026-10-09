#!/usr/bin/env python3
"""brain.py - the base-station decision layer for DHWANI.

WHAT IT DECIDES. Once a second it looks at the rover's telemetry (pose, mission state, recent
events, link quality) and picks ONE of four commands: CONTINUE, GOTO_EVENT, HOME, STOP. That is
the whole vocabulary. The rover's own mission node executes the command; move_base plans the
path; the dead-man's permit stays in base_station.py. This file only decides.

THREE BACKENDS, one question. `rules` is a deterministic table and is always loaded. `laya` is
the local non-autoregressive decision model (convaiinnovations/laya, runs on this Mac without
internet). `jev` is TypeSafe's hosted equivalent (needs TYPESAFE_API_KEY and a network). The
model backends answer the same typed `choice` question with a calibrated confidence; below
--min-conf the rules decide instead and the row is marked GATED. Measured on 24 Sep 2026 the
untuned Laya checkpoint answered GOTO_EVENT for every scenario at 0.09-0.21 confidence, so the
gate is not decoration - it is what keeps an untuned model from driving the rover.

TWO TRANSPORTS. --ros talks rosbridge over Wi-Fi or the USB link (works today); --lora reads the
"R,x,y,yaw,state" and "E,n,kind,x,y" lines the rover sends over LoRa and answers "C,<cmd>"
(needs the CP2102 adapter on the base station). --replay runs five built-in scenarios and is how
the backends are compared on identical inputs.

NOTHING MOVES unless --act is given. The default prints what it would send.
"""
import argparse
import json
import math
import os
import sys
import time

COMMANDS = ["CONTINUE", "GOTO_EVENT", "HOME", "STOP"]
QUESTION = {
    "next": {
        "type": "choice",
        "instructions": ("You are the base-station controller of a search and rescue ground rover working without GPS. "
                         "From the telemetry, choose the single next command."),
        "criteria": {
            "CONTINUE": "no unvisited person event, battery_pct above 20, lora_link ok, state not "
                        "STUCK: keep the current mission going",
            "GOTO_EVENT": "events contains a person that is not yet visited: drive to it and confirm",
            "HOME": "battery_pct below 20, or lora_link lost, or the mission is complete: return "
                    "to the start point",
            "STOP": "state is STUCK, or a tilt event exists, or a person is closer than 1 m: halt "
                    "immediately",
        },
    },
    "risk": {"type": "score", "instructions": "How risky is the rover's situation right now?",
             "criteria": ["safe", "caution", "danger"]},
}


# ---------------------------------------------------------------- backends
class Rules(object):
    name = "rules"

    def decide(self, st):
        ev = st["events"]
        person = [e for e in ev if e.get("kind") == "person" and not e.get("visited")]
        near = [e for e in person if e.get("distance_m", 9) < 1.0]
        if st["rover"]["state"] == "STUCK" or any(e.get("kind") == "tilt" for e in ev) or near:
            return "STOP", 1.0
        if st["rover"]["battery_pct"] is not None and st["rover"]["battery_pct"] < 20:
            return "HOME", 1.0
        if st["rover"]["lora_link"] == "lost":
            return "HOME", 1.0
        if person:
            return "GOTO_EVENT", 1.0
        return "CONTINUE", 1.0


class Laya(object):
    name = "laya"

    def __init__(self, device):
        from laya import Router
        t = time.time()
        self.r = Router(preload=False, device=device)
        self.r.preload(["english"])
        self.load_s = time.time() - t

    def decide(self, st):
        a = self.r.predict(st, QUESTION)["answers"]
        return a["next"]["choice"], float(a["next"]["confidence"])


class Jev(object):
    name = "jev"

    def __init__(self):
        from typesafe_sdk import TypeSafeClient, Choice, Score
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("jev backend needs TYPESAFE_API_KEY in the environment")
        self.c = TypeSafeClient()
        q = QUESTION["next"]
        self.q = {"next": Choice(criteria=q["criteria"], instructions=q["instructions"]),
                  "risk": Score(criteria=QUESTION["risk"]["criteria"],
                                instructions=QUESTION["risk"]["instructions"])}
        self.load_s = 0.0

    def decide(self, st):
        a = self.c.system_one(state=st, questions=self.q).answers["next"]
        return a.choice, float(getattr(a, "confidence", 0.0) or 0.0)


# ---------------------------------------------------------------- transports
class Replay(object):
    """Five scenarios. Same inputs for every backend, so the comparison is fair."""
    def __init__(self):
        def st(x, y, yaw, state, bat, events, link):
            return {"rover": {"x": x, "y": y, "yaw_deg": yaw, "state": state, "battery_pct": bat,
                              "lora_link": link}, "events": events,
                    "mission": "tour the hall, record any person, come back"}
        self.cases = [
            ("quiet tour", st(3.2, 1.1, 40, "TOUR", 82, [], "ok"), "CONTINUE"),
            ("person seen", st(7.9, 2.3, 95, "TOUR", 76,
                               [{"kind": "person", "x": 9.1, "y": 2.0, "age_s": 2,
                                 "distance_m": 1.2, "visited": False}], "ok"), "GOTO_EVENT"),
            ("battery low", st(11.0, 4.0, 180, "TOUR", 14, [], "weak"), "HOME"),
            ("tilted/stuck", st(6.1, 0.4, 12, "STUCK", 60, [{"kind": "tilt", "deg": 28}], "ok"), "STOP"),
            ("link lost", st(15.5, 6.0, 200, "GOTO", 55, [], "lost"), "HOME"),
        ]
        self.i = 0

    def state(self):
        if self.i >= len(self.cases):
            return None
        name, st, want = self.cases[self.i]
        self.i += 1
        return name, st, want

    def send(self, cmd):
        pass


class Ros(object):
    """rosbridge JSON over a websocket. No ROS install needed on the laptop."""
    def __init__(self, url):
        from websocket import create_connection
        self.ws = create_connection(url, timeout=3)
        for topic, typ in (("/odom", "nav_msgs/Odometry"), ("/mission/state", "std_msgs/String"),
                           ("/events", "std_msgs/String"), ("/mission/link", "std_msgs/String")):
            self.ws.send(json.dumps({"op": "subscribe", "topic": topic, "type": typ,
                                     "throttle_rate": 200}))
        self.ws.send(json.dumps({"op": "advertise", "topic": "/mission/command",
                                 "type": "std_msgs/String"}))
        nan = float("nan")
        self.pose, self.mstate, self.events, self.link = (nan, nan, nan), "UNKNOWN", [], "wifi"
        self.t0 = time.time()

    def pump(self, seconds):
        end = time.time() + seconds
        self.ws.settimeout(0.2)
        while time.time() < end:
            try:
                m = json.loads(self.ws.recv())
            except Exception:
                continue
            if m.get("op") != "publish":
                continue
            t, msg = m["topic"], m["msg"]
            if t == "/odom":
                p, q = msg["pose"]["pose"]["position"], msg["pose"]["pose"]["orientation"]
                yaw = math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]),
                                 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))
                self.pose = (p["x"], p["y"], math.degrees(yaw))
            elif t == "/mission/state":
                self.mstate = msg["data"]
            elif t == "/events":
                try:
                    e = json.loads(msg["data"])
                    e["age_s"] = 0
                    self.events = (self.events + [e])[-5:]
                except ValueError:
                    pass
            elif t == "/mission/link":
                self.link = msg["data"]

    def state(self):
        self.pump(1.0)
        x, y, yaw = self.pose
        for e in self.events:
            e["age_s"] = e.get("age_s", 0) + 1
        r2 = lambda v: None if v != v else round(v, 2)
        return "live", {"rover": {"x": r2(x), "y": r2(y), "yaw_deg": r2(yaw),
                                  "state": self.mstate, "battery_pct": None,
                                  "lora_link": self.link},
                        "events": self.events,
                        "mission": "tour the hall, record any person, come back"}, None

    def send(self, cmd):
        self.ws.send(json.dumps({"op": "publish", "topic": "/mission/command",
                                 "msg": {"data": cmd}}))


class Lora(object):
    """The LoRa control plane: R, and E, lines in; C,<cmd> out. Same grammar as the rover."""
    def __init__(self, port, baud=9600):
        import serial
        self.s = serial.Serial(port, baud, timeout=0.2)
        self.pose, self.mstate, self.events, self.last_rx = (0.0, 0.0, 0.0), "UNKNOWN", [], 0.0

    def state(self):
        end = time.time() + 1.0
        while time.time() < end:
            ln = self.s.readline().decode("ascii", "ignore").strip()
            if not ln:
                continue
            f = ln.split(",")
            if f[0] == "R" and len(f) >= 5:
                self.pose = (float(f[1]), float(f[2]), float(f[3]))
                self.mstate = f[4]
                self.last_rx = time.time()
            elif f[0] == "E" and len(f) >= 5:
                self.events = (self.events + [{"kind": f[2], "x": float(f[3]), "y": float(f[4]),
                                               "age_s": 0, "visited": False}])[-5:]
        age = time.time() - self.last_rx
        link = "ok" if age < 5 else ("weak" if age < 15 else "lost")
        x, y, yaw = self.pose
        return "lora", {"rover": {"x": x, "y": y, "yaw_deg": yaw, "state": self.mstate,
                                  "battery_pct": None, "lora_link": link},
                        "events": self.events,
                        "mission": "tour the hall, record any person, come back"}, None

    def send(self, cmd):
        self.s.write(("C,%s\n" % cmd).encode("ascii"))


# ---------------------------------------------------------------- main loop
def to_command(decision, st):
    if decision == "GOTO_EVENT":
        ev = [e for e in st["events"] if e.get("kind") == "person"]
        if not ev:
            return None
        return "GOTO %.2f %.2f" % (ev[-1]["x"], ev[-1]["y"])
    return {"HOME": "HOME", "STOP": "STOP"}.get(decision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rules", choices=["rules", "laya", "jev"])
    ap.add_argument("--device", default="mps", help="laya only: mps | cpu | cuda")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--ros", help="rosbridge url, e.g. ws://192.168.55.1:9090")
    ap.add_argument("--lora", help="serial port of the base-station E32")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--act", action="store_true", help="really send commands")
    ap.add_argument("--json", help="append every decision row to this file")
    ap.add_argument("--env-file", help="KEY=VALUE file to load TYPESAFE_API_KEY from (never printed)")
    args = ap.parse_args()
    if args.env_file:
        for ln in open(os.path.expanduser(args.env_file)):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    rules = Rules()
    if args.backend == "laya":
        model = Laya(args.device)
    elif args.backend == "jev":
        model = Jev()
    else:
        model = None
    if model is not None:
        print("%s loaded in %.1f s" % (model.name, model.load_s))
    tr = Replay() if args.replay else (Ros(args.ros) if args.ros else Lora(args.lora))
    print("%-13s %-11s %-5s %-7s %-6s %-14s %s" % ("scenario", "decision", "conf", "ms", "gated",
                                                   "expected", "would send"))
    t_end = time.time() + args.seconds
    rows, ok, n = [], 0, 0
    while time.time() < t_end or args.replay:
        s = tr.state()
        if s is None:
            break
        name, st, want = s
        t = time.time()
        if model is not None:
            try:
                dec, conf = model.decide(st)
            except Exception as e:
                dec, conf = "ERROR:%s" % type(e).__name__, 0.0
        else:
            dec, conf = rules.decide(st)
        ms = (time.time() - t) * 1000
        gated = model is not None and conf < args.min_conf
        final = rules.decide(st)[0] if gated else dec
        cmd = to_command(final, st)
        if want is not None:
            n += 1
            ok += int(final == want)
        row = {"scenario": name, "backend": args.backend, "raw": dec, "conf": round(conf, 3),
               "ms": round(ms, 1), "gated": gated, "final": final, "expected": want,
               "command": cmd, "sent": bool(cmd and args.act), "t": time.time()}
        rows.append(row)
        print("%-13s %-11s %.2f  %6.0f  %-6s %-14s %s" % (
            name, dec if len(dec) < 12 else dec[:11], conf, ms, "GATED" if gated else "",
            want or "-", (cmd or "-") + ("  [SENT]" if row["sent"] else "  [dry-run]" if cmd else "")))
        if cmd and args.act:
            tr.send(cmd)
    if n:
        print("final decisions correct: %d/%d (%s%s)" % (ok, n, args.backend,
                                                          " + rules gate" if model else ""))
    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
