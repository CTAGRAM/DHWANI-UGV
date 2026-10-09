#!/usr/bin/env python3
"""Send DHWANI a plain-English mission through the remote dashboard and record the OAK-D camera with
the object boxes drawn on it, until the mission ends.

It talks to gateway.py exactly like the dashboard page does (PIN, take control, parse, start,
ARM), so every safety rule of the gateway still applies: STOP on the page works for anyone, and
because this script holds control, the gateway's dead-man permit stops if the script dies and the
rover halts within 6 s. Ctrl-C sends STOP and disarms.

    DHWANI_PIN=... python3 map_and_record.py --host 10.42.0.1 \
        --text "Map this whole room and come back when done" --max-min 8

Writes <out>/camera_overlay.mp4 (10 fps, real time), <out>/log.jsonl (every frame's detections,
pose, executive state and events) and <out>/summary.json.
"""
import argparse
import asyncio
import collections
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import websockets

W, H, HUD = 832, 468, 96          # OAK preview 416x234 shown at 2x, plus a status bar
FPS = 10
DET_STALE = 0.7                   # s: older boxes are not drawn


def colour(label):
    if label == "person":
        return (87, 77, 255)      # BGR red, as on the dashboard
    h = 0
    for ch in label:
        h = (h * 31 + ord(ch)) % 360
    hsv = np.uint8([[[h // 2, 200, 250]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class Recorder(object):
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.raw = os.path.join(out_dir, "camera_overlay_raw.mp4")
        self.vw = cv2.VideoWriter(self.raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H + HUD))
        self.lock = threading.Lock()
        self.frame = None                 # latest decoded OAK frame (BGR, W x H)
        self.frame_t = 0.0
        self.dets, self.dets_t = [], 0.0
        self.status = {"phase": "?", "decision": "", "mission": "", "v": 0.0}
        self.seen = collections.Counter()
        self.n_frames = self.n_written = 0
        self.t0 = time.time()
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def on_jpeg(self, jpeg):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        with self.lock:
            self.frame, self.frame_t = img, time.time()
            self.n_frames += 1

    def on_dets(self, objs):
        with self.lock:
            self.dets, self.dets_t = objs, time.time()
            for o in objs:
                self.seen[o["label"]] += 1

    def compose(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            dets = list(self.dets) if time.time() - self.dets_t < DET_STALE else []
            st = dict(self.status)
        canvas = np.zeros((H + HUD, W, 3), np.uint8)
        if frame is not None:
            canvas[:H] = frame
        else:
            cv2.putText(canvas, "waiting for the OAK-D camera", (W // 2 - 170, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        count = collections.Counter()
        for o in dets:
            b = o.get("box") or [0, 0, 0, 0]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            col = colour(o["label"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
            txt = "%s %d%%" % (o["label"], round(100 * o["conf"])) + ("  %.1f m" % o["z"] if o.get("z") else "")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 2
            cv2.rectangle(canvas, (x1, ty), (x1 + tw + 6, ty + th + 6), col, -1)
            cv2.putText(canvas, txt, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (19, 14, 11), 1, cv2.LINE_AA)
            count[o["label"]] += 1
        # status bar
        el = time.time() - self.t0
        y = H + 22
        line1 = "DHWANI  %02d:%02d  mission %s  phase %s  speed %.2f m/s" % (
            el // 60, el % 60, st.get("mission") or "-", st.get("phase") or "?", st.get("v") or 0.0)
        cv2.putText(canvas, line1, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, ("decision: " + (st.get("decision") or "-"))[:110], (10, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1, cv2.LINE_AA)
        now = ", ".join("%s %d" % (k, v) for k, v in sorted(count.items())) or "nothing"
        cv2.putText(canvas, ("seeing now: " + now)[:110], (10, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 230, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, "OAK-D on-chip YOLOv6-nano, live boxes", (W - 330, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return canvas

    def run(self):
        nxt = time.time()
        while not self.stop.is_set():
            self.vw.write(self.compose())
            self.n_written += 1
            nxt += 1.0 / FPS
            time.sleep(max(0.0, nxt - time.time()))

    def close(self):
        self.stop.set()
        self.th.join(timeout=2)
        self.vw.release()
        final = os.path.join(self.out_dir, "camera_overlay.mp4")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", self.raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "23", "-movflags", "+faststart", final])
        if r.returncode == 0:
            os.remove(self.raw)
            return final
        return self.raw


async def main(a):
    os.makedirs(a.out, exist_ok=True)
    rec = Recorder(a.out)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    st = {"you": None, "token": None, "controller": None, "halted": False, "parsed": None, "acks": [],
          "exec": {}, "mission_state": None, "was_active": False, "events": [], "pose": None}
    t_start = time.time()
    url = "ws://%s:%d/ws" % (a.host, a.port)
    print("connecting", url, flush=True)
    async with websockets.connect(url, max_size=2 ** 22, ping_interval=10, ping_timeout=20) as ws:

        async def send(o):
            if st["token"] and "token" not in o:
                o["token"] = st["token"]
            await ws.send(json.dumps(o))

        async def reader():
            async for msg in ws:
                if isinstance(msg, bytes):
                    if msg[:1] == b"J" and msg[1] == 1:
                        rec.on_jpeg(msg[2:])
                    continue
                d = json.loads(msg)
                t = d.get("t")
                if t == "hello":
                    st["you"] = d.get("you")
                elif t == "auth":
                    st["token"] = d.get("token") if d.get("ok") else None
                    print("auth", "ok" if d.get("ok") else "FAILED: %s" % d.get("msg"), flush=True)
                elif t == "control":
                    st["controller"] = d.get("controller_client")
                    st["halted"] = bool(d.get("halted"))
                elif t == "parsed":
                    st["parsed"] = d.get("mission")
                elif t == "ack":
                    st["acks"].append(d)
                    print("ack", d.get("cmd"), "ok" if d.get("ok") else "REFUSED", "-", d.get("msg"), flush=True)
                elif t == "dets":
                    rec.on_dets(d.get("objects") or [])
                    log.write(json.dumps({"rx": round(time.time(), 3), "dets": d.get("objects")}) + "\n")
                elif t == "exec":
                    ex = d.get("state") or {}
                    st["exec"] = ex
                    dec = ex.get("decision") or {}
                    with rec.lock:
                        rec.status["phase"] = ex.get("phase")
                        rec.status["decision"] = dec.get("reason") or ex.get("reason") or ""
                    if ex.get("phase") in ("ACTIVE", "DWELL"):
                        st["was_active"] = True
                elif t == "mission_state":
                    st["mission_state"] = d.get("state")
                elif t == "pose":
                    st["pose"] = d
                    with rec.lock:
                        rec.status["v"] = d.get("v", 0.0)
                elif t == "event":
                    ev = d.get("event") or {}
                    st["events"].append(ev)
                    log.write(json.dumps({"rx": round(time.time(), 3), "event": ev}) + "\n")
                    print("event", ev.get("kind"), ev.get("text"), flush=True)
                elif t == "toast":
                    print("toast", d.get("msg"), flush=True)

        async def wait(cond, timeout, what):
            t0 = time.time()
            while not cond():
                if rtask.done():
                    raise RuntimeError("lost the connection to the rover while waiting for " + what)
                if time.time() - t0 > timeout:
                    raise RuntimeError("timed out waiting for " + what)
                await asyncio.sleep(0.1)

        async def pinger():
            while True:
                await send({"t": "ping", "ts": time.time() * 1000, "rtt": None})
                await asyncio.sleep(2.0)

        rtask = asyncio.ensure_future(reader())
        ptask = asyncio.ensure_future(pinger())
        try:
            await wait(lambda: st["you"], 10, "hello")
            await send({"t": "camera", "on": True})
            await send({"t": "auth", "name": a.name, "pin": a.pin})
            await wait(lambda: st["token"], 10, "PIN accepted")
            await send({"t": "take_control", "force": True})
            await wait(lambda: st["controller"] == st["you"], 60, "control")
            print("in control", flush=True)
            if st["halted"]:
                await send({"t": "resume"})
                await asyncio.sleep(1.0)
            for attempt in range(6):          # a slow or dead viewer can hold up the gateway's queue
                await send({"t": "parse", "text": a.text, "req": 1})
                try:
                    await wait(lambda: st["parsed"], 10, "parse")
                    break
                except RuntimeError:
                    print("no parse reply yet, asking again", flush=True)
            if not st["parsed"]:
                raise RuntimeError("the gateway never answered the parse request")
            m = st["parsed"]
            print("parsed:", json.dumps(m), flush=True)
            if not m.get("understood") or not any(k in (m.get("task") or "") for k in ("MAP", "SEARCH", "EXPLORE")):
                raise RuntimeError("not an exploration mission, not starting: %s" % m.get("task"))
            with rec.lock:
                rec.status["mission"] = m.get("task")
            n_ack = len(st["acks"])
            await send({"t": "start_mission", "mission": m})
            await wait(lambda: len(st["acks"]) > n_ack, 60, "mission ack")
            ack = st["acks"][-1]
            if not ack.get("ok"):
                raise RuntimeError("mission refused: %s" % ack.get("msg"))
            if ack.get("need_arm"):
                for attempt in range(4):
                    await send({"t": "arm", "on": True})
                    try:
                        await wait(lambda: (st["exec"] or {}).get("armed"), 15, "executive armed")
                        break
                    except RuntimeError:
                        print("executive not armed yet, sending ARM again", flush=True)
                if not (st["exec"] or {}).get("armed"):
                    raise RuntimeError("the executive did not arm")
                print("ARMED: the rover drives itself now", flush=True)
            deadline = time.time() + a.max_min * 60
            while True:
                if rtask.done():
                    print("connection to the rover lost (battery? Wi-Fi?): saving the video", flush=True)
                    break
                ph = (st["exec"] or {}).get("phase")
                if st["was_active"] and ph in ("DONE", "IDLE", "HALTED", "READY"):
                    print("mission ended, phase", ph, flush=True)
                    break
                if time.time() > deadline:
                    print("time limit reached: STOP", flush=True)
                    await send({"t": "stop", "why": "map_and_record time limit"})
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(3.0)                   # a few seconds of the rover at rest
        except (KeyboardInterrupt, asyncio.CancelledError):
            await send({"t": "stop", "why": "map_and_record interrupted"})
            raise
        finally:
            try:
                await send({"t": "arm", "on": False})
                await send({"t": "release"})
            except Exception:
                pass
            ptask.cancel(); rtask.cancel()
            final = rec.close()
            log.close()
            ex = st["exec"] or {}
            summary = {"text": a.text, "mission": st["parsed"], "started": time.strftime("%Y-%m-%d %H:%M:%S",
                       time.localtime(t_start)), "seconds": round(time.time() - t_start, 1),
                       "final_phase": ex.get("phase"), "odo_m": ex.get("odo_m"), "persons": ex.get("persons"),
                       "frames_received": rec.n_frames, "frames_written": rec.n_written,
                       "objects_seen_counts": dict(rec.seen), "video": final,
                       "events": [e.get("kind") + ": " + (e.get("text") or "") for e in st["events"]]}
            json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
            print("video:", final, flush=True)
            print("summary:", json.dumps({k: summary[k] for k in ("seconds", "final_phase", "odo_m",
                                                                  "objects_seen_counts")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.42.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--pin", default=os.environ.get("DHWANI_PIN"))
    ap.add_argument("--name", default="map-recorder")
    ap.add_argument("--text", default="Map this whole room and come back when done")
    ap.add_argument("--max-min", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.expanduser(time.strftime("~/dhwani_runs/%Y-%m-%d_%H%M_map_room")))
    a = ap.parse_args()
    if not a.pin:
        sys.exit("set DHWANI_PIN or pass --pin")
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        print("interrupted: STOP sent", flush=True)
