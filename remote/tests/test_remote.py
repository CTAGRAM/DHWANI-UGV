#!/usr/bin/env python3
"""End-to-end test of the DHWANI remote dashboard through its public Cloudflare link.

Runs on the Mac against a gateway started with --dry-run, so every motion command lands on a
*_dryrun topic and the rover cannot move. What reached the rover is checked with rostopic on the
rover itself (over ssh, see DHWANI_ROVER_SSH below), so this tests the whole path: browser protocol ->
tunnel -> gateway -> ROS.

    python3 test_remote.py https://<name>.trycloudflare.com/ <test-pin>
"""
import asyncio
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

from websockets.asyncio.client import connect

# how to run a command on the rover, e.g. DHWANI_ROVER_SSH="ssh jetson@10.42.0.1"
ROVER = shlex.split(os.environ.get("DHWANI_ROVER_SSH", "ssh jetson@10.42.0.1"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ("  (" + detail + ")" if detail else ""))


def rover(cmd, timeout=90):
    return subprocess.run(ROVER + [cmd], capture_output=True, text=True, timeout=timeout).stdout


def start_recorders(topics, seconds=60):
    cmd = "pkill -x rostopic; source /opt/ros/melodic/setup.bash; rm -f /tmp/t_*.txt; touch /tmp/t_marker; "
    for t in topics:
        cmd += "nohup timeout %d rostopic echo %s > /tmp/t%s.txt 2>&1 & " % (seconds, t, t.replace("/", "_"))
    cmd += "sleep 14; echo ready"
    out = rover(cmd)
    if "ready" not in out:
        raise SystemExit("could not start the recorders on the rover: %r" % out[-200:])


def read_topic(t):
    f = "/tmp/t%s.txt" % t.replace("/", "_")
    txt = rover("[ %s -nt /tmp/t_marker ] || [ -s %s ] && cat %s || echo NO_FILE" % (f, f, f))
    if "NO_FILE" in txt:
        raise SystemExit("recorder output missing for %s" % t)
    return [m for m in txt.split("---") if m.strip()]


def twists(msgs):
    out = []
    for m in msgs:
        lx = re.search(r"linear:\s*\n\s*x: ([-0-9.e]+)", m)
        az = re.search(r"angular:\s*\n\s*x: [-0-9.e]+\s*\n\s*y: [-0-9.e]+\s*\n\s*z: ([-0-9.e]+)", m)
        if lx and az:
            out.append((round(float(lx.group(1)), 3), round(float(az.group(1)), 3)))
    return out


class Client(object):
    def __init__(self, url):
        self.url = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"
        self.inbox = []
        self.token = None

    async def __aenter__(self):
        self.ws = await connect(self.url, max_size=2 ** 22)
        self.reader = asyncio.ensure_future(self.read())
        return self

    async def __aexit__(self, *a):
        self.reader.cancel()
        await self.ws.close()

    async def read(self):
        async for m in self.ws:
            if isinstance(m, bytes):
                self.inbox.append({"t": "_camera", "n": len(m), "src": m[1], "at": time.time()})
            else:
                d = json.loads(m)
                d["_at"] = time.time()
                self.inbox.append(d)

    async def send(self, d):
        if self.token and "token" not in d:
            d["token"] = self.token
        await self.ws.send(json.dumps(d))

    async def wait(self, t, pred=lambda d: True, timeout=8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for d in self.inbox:
                if d.get("t") == t and pred(d):
                    self.inbox.remove(d)
                    return d
            await asyncio.sleep(0.02)
        return None

    def drain(self, t):
        self.inbox = [d for d in self.inbox if d.get("t") != t]


async def rtt(c, n=20):
    xs = []
    for i in range(n):
        ts = time.time()
        await c.send({"t": "ping", "ts": ts})
        d = await c.wait("pong", lambda d: d.get("ts") == ts, 5)
        if d:
            xs.append((d["_at"] - ts) * 1000.0)
        await asyncio.sleep(0.15)
    return xs


async def main(url, pin):
    async with Client(url) as a:
        hello = await a.wait("hello")
        check("hello received", hello is not None, "dry_run=%s parser=%s" % (hello and hello.get("dry_run"), hello and hello.get("parser")))
        if not hello or not hello.get("dry_run"):
            print("REFUSING to continue: gateway is not in dry-run mode")
            return
        check("map arrives", await a.wait("map", timeout=10) is not None)
        await asyncio.sleep(3)
        poses = [d for d in a.inbox if d.get("t") == "pose"]
        scans = [d for d in a.inbox if d.get("t") == "scan"]
        check("pose stream ~10 Hz", len(poses) >= 20, "%d poses in 3 s" % len(poses))
        check("lidar stream ~3 Hz", len(scans) >= 6, "%d scans in 3 s, %d points each" % (len(scans), len(scans[-1]["pts"]) if scans else 0))
        st = await a.wait("stats", timeout=3)
        check("stats", st is not None, st and "cpu %s%%, ram %s MB, lidar %s Hz, temp %s C" % (st.get("cpu"), (st.get("mem") or {}).get("used_mb"), st.get("scan_hz"), st.get("temp")))

        r0 = await rtt(a)
        check("round trip without camera", len(r0) >= 15, "median %.0f ms, p95 %.0f ms" % (statistics.median(r0), sorted(r0)[int(0.95 * len(r0)) - 1]))
        await a.send({"t": "camera", "on": True})
        await asyncio.sleep(5)
        cams = [d for d in a.inbox if d.get("t") == "_camera"]
        fps = len(cams) / 5.0
        check("camera frames arrive", len(cams) >= 5, "%.1f fps, %d bytes avg" % (fps, sum(d["n"] for d in cams) / max(1, len(cams))))
        r1 = await rtt(a)
        check("round trip with camera on", len(r1) >= 15, "median %.0f ms, p95 %.0f ms" % (statistics.median(r1), sorted(r1)[int(0.95 * len(r1)) - 1]))
        await a.send({"t": "camera", "on": False})
        a.drain("_camera")
        await a.send({"t": "depth", "on": True})             # needs the OAK-D streaming
        await asyncio.sleep(5)
        deps = [d for d in a.inbox if d.get("t") == "_camera" and d.get("src") == 2]
        check("depth frames arrive", len(deps) >= 10, "%.1f fps, %d bytes avg" % (len(deps) / 5.0, sum(d["n"] for d in deps) / max(1, len(deps))))
        await a.send({"t": "depth", "on": False})
        await asyncio.sleep(1)
        a.drain("_camera")

        # ---- auth
        await a.send({"t": "auth", "pin": "000000", "name": "tester"})
        d = await a.wait("auth")
        check("wrong PIN refused", d and not d.get("ok"))
        await a.send({"t": "drive", "v": 1, "w": 0, "token": "nonsense"})
        d = await a.wait("ack")
        check("drive with a fake token refused", d and not d.get("ok"), d and d.get("msg"))
        await a.send({"t": "auth", "pin": pin, "name": "tester"})
        d = await a.wait("auth")
        check("test PIN accepted", d and d.get("ok"))
        a.token = d and d.get("token")
        await a.send({"t": "drive", "v": 1, "w": 0})
        d = await a.wait("ack")
        check("drive before Take control refused", d and not d.get("ok"), d and d.get("msg"))

        # ---- hold-to-drive and the 0.5 s cut-off, verified on the rover
        # run the slow ssh setup in a thread so this client keeps answering the server's pings
        await asyncio.get_running_loop().run_in_executor(
            None, start_recorders, ["/dashboard/cmd_vel_dryrun", "/mission/command_dryrun", "/move_base/cancel_dryrun"])
        await a.send({"t": "take_control"})
        d = await a.wait("control", lambda d: d.get("controller") == "tester")
        check("take control", d is not None)
        await a.send({"t": "resume"})      # a STOP from an earlier run holds until the controller resumes
        await asyncio.sleep(0.3)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            await a.send({"t": "drive", "v": 0.5, "w": -0.25})
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.5)          # stop sending: the gateway must zero and go quiet
        await a.send({"t": "stop", "why": "automated test"})
        await asyncio.sleep(0.5)
        await a.send({"t": "drive", "v": 0.5, "w": 0})
        d = await a.wait("ack", lambda d: d.get("cmd") == "drive")
        check("drive refused after STOP until Resume", d and not d.get("ok") and d.get("halted"), d and d.get("msg"))
        await a.send({"t": "resume"})
        await asyncio.sleep(0.3)
        await a.send({"t": "drive", "v": -0.2, "w": 0})
        await asyncio.sleep(1.2)
        # ---- missions without the executive running
        await a.send({"t": "start_mission", "mission": {"task": "SEARCH_SURVIVORS", "understood": True, "text": "find people"}})
        d = await a.wait("ack", lambda d: d.get("cmd") == "start_mission")
        check("search mission refused while executive is off", d and not d.get("ok"), d and d.get("msg"))
        await a.send({"t": "start_mission", "mission": {"task": "GOTO_WAYPOINT", "waypoint": "A", "understood": True, "text": "go to A"}})
        d = await a.wait("ack", lambda d: d.get("cmd") == "start_mission")
        check("go to waypoint A accepted", d and d.get("ok"), d and d.get("msg"))
        await a.send({"t": "start_mission", "mission": {"task": "SELF_DESTRUCT", "understood": True}})
        d = await a.wait("ack", lambda d: d.get("cmd") == "start_mission")
        check("unknown task refused, not turned into a mission", d and not d.get("ok"), d and d.get("msg"))

        # ---- a viewer with no PIN can always STOP
        async with Client(url) as b:
            await b.wait("hello")
            await b.send({"t": "stop", "why": "viewer test"})
            d = await b.wait("toast", lambda d: "STOP" in d.get("msg", ""))
            check("viewer without PIN can STOP", d is not None)
        # ---- parse through the gateway
        await a.send({"t": "parse", "text": "go through this mine and find any person alive, come back if the link drops", "req": 7})
        d = await a.wait("parsed")
        m = d and d.get("mission") or {}
        check("parse answers", bool(m), "task=%s understood=%s parser=%s %s ms" % (m.get("task"), m.get("understood"), m.get("parser"), m.get("parse_ms")))

    # ---- controller disconnect releases the lock
    async with Client(url) as c:
        d = await c.wait("control", timeout=4)
        check("lock released after the controller disconnected", d and d.get("controller") is None, "controller=%s" % (d and d.get("controller")))

    # ---- what actually reached the rover
    tw = twists(read_topic("/dashboard/cmd_vel_dryrun"))
    runs = []
    for v in tw:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    print("cmd_vel_dryrun runs:", runs)
    drive = [n for v, n in runs if v == (0.15, -0.3)]
    check("hold-to-drive published at 20 Hz while held", drive and drive[0] >= 25, "%s messages of (0.15, -0.30)" % (drive[0] if drive else 0))
    after = runs[runs.index([(0.15, -0.3), drive[0]]) + 1] if drive else None
    check("zeros after the command went stale, then silence", after and after[0] == (0.0, 0.0) and after[1] <= 15,
          "%s" % after)
    check("reverse after Resume published", any(v == (-0.06, 0.0) for v, n in runs))
    cmds = read_topic("/mission/command_dryrun")
    data = [re.search(r'data: "(.*)"', m).group(1) for m in cmds if re.search(r'data: "(.*)"', m)]
    check("mission commands reached the rover", "STOP" in data and "GOTO A" in data, str(data))
    check("goal cancel on STOP", len(read_topic("/move_base/cancel_dryrun")) >= 1)
    log = rover("tail -12 ~/dhwani_remote/logs/commands.log")
    check("audit log records the test", "tester" in log and "STOP" in log)

    n_ok = sum(1 for r in RESULTS if r[1])
    print("\n%d / %d checks passed" % (n_ok, len(RESULTS)))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run.json"), "w") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "url": url,
                   "results": [{"check": a, "ok": b, "detail": c} for a, b, c in RESULTS]}, f, indent=1)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
