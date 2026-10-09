#!/usr/bin/env python3
"""DHWANI status on the expansion board's 0.91 inch OLED (SSD1306, 128 x 32, I2C bus 1, 0x3C).

At power-on it shows "DHWANI". Then, once a second:
  line 1  DHWANI, a Wi-Fi dot (filled = joined to a Wi-Fi network, ring = only the rover's own
          DHWANI hotspot), battery icon and charge left
  line 2  pack voltage and distance driven (the executive's figure, as on the dashboard)
  line 3  every 3 s the next of: the rover's IP address, the mission state, people found

The battery is read straight from the INA219 (0x41), so it shows before ROS is up; the curve and
the smoothing are the executive's, so the two agree. Everything else comes from /executive/state.
Started at boot by the jetson user's crontab (@reboot start.sh); no sudo, no extra packages
(Pillow 5.1 and rospy are already on the rover). The I2C writes use their own file descriptor; the
kernel serialises them with the executive's INA219 reads.

  python3 oled.py              run on the rover
  python3 oled.py --preview D  write sample screens as PNG files into folder D (no hardware)
"""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time

from PIL import Image, ImageDraw, ImageFont

BUS, OLED_ADDR, INA_ADDR = 1, 0x3C, "0x41"
W, H = 128, 32
I2C_SLAVE = 0x0703
FONT_DIR = os.environ.get("DHWANI_FONTS", "/usr/share/fonts/truetype/dejavu")
# same as executive_node.py: typical Li-ion 18650 per cell, not calibrated on these cells
CELLS = 3
CELL_CURVE = ((3.40, 0.0), (3.50, 12.0), (3.60, 25.0), (3.70, 40.0), (3.80, 55.0), (3.90, 68.0),
              (4.00, 80.0), (4.10, 90.0), (4.20, 100.0))
BATTERY_READ_S, BATTERY_TAU_S = 5.0, 30.0
SPLASH_S = 4.0
ROTATE_S = 3.0
STATE_STALE_S = 5.0


def log(msg):
    print("[oled %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def pack_pct(volts):
    v = volts / CELLS
    if v <= CELL_CURVE[0][0]:
        return 0.0
    for (v0, p0), (v1, p1) in zip(CELL_CURVE, CELL_CURVE[1:]):
        if v <= v1:
            return p0 + (p1 - p0) * (v - v0) / (v1 - v0)
    return 100.0


def pack_volts():
    out = subprocess.check_output(["i2ctransfer", "-y", str(BUS), "w1@" + INA_ADDR, "0x02", "r2"],
                                  stderr=subprocess.STDOUT, timeout=2)
    hi, lo = [int(b, 16) for b in out.split()]
    raw = (hi << 8) | lo
    if raw & 1:
        raise ValueError("INA219 bus voltage overflow")
    v = (raw >> 3) * 0.004
    if not 6.0 < v < 14.0:
        raise ValueError("implausible pack voltage %.2f V" % v)
    return v


def wifi():
    """(client IP or None, hotspot up)."""
    ip = None
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show", "wlan1"], timeout=2).decode()
        parts = out.split()
        if "inet" in parts:
            ip = parts[parts.index("inet") + 1].split("/")[0]
    except Exception:
        pass
    try:
        with open("/sys/class/net/wlan0/operstate") as f:
            ap = f.read().strip() == "up"
    except OSError:
        ap = False
    return ip, ap


class Oled(object):
    """Just enough SSD1306: init, then whole 128 x 32 frames in horizontal addressing mode."""

    def __init__(self):
        self.fd = os.open("/dev/i2c-%d" % BUS, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, OLED_ADDR)
        self.cmd(0xAE, 0xD5, 0x80, 0xA8, 0x1F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00,
                 0xA1, 0xC8, 0xDA, 0x02, 0x81, 0x8F, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0x2E, 0xAF)

    def cmd(self, *codes):
        for c in codes:
            os.write(self.fd, bytes([0x00, c]))

    def show(self, img):
        px = img.load()
        buf = bytearray(W * H // 8)
        for page in range(H // 8):
            for x in range(W):
                b = 0
                for bit in range(8):
                    if px[x, page * 8 + bit]:
                        b |= 1 << bit
                buf[page * W + x] = b
        self.cmd(0x21, 0, W - 1, 0x22, 0, H // 8 - 1)
        for i in range(0, len(buf), 16):
            os.write(self.fd, bytes([0x40]) + bytes(buf[i:i + 16]))

    def close(self):
        try:
            self.cmd(0xAE)
        finally:
            os.close(self.fd)


def fonts():
    try:
        return (ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 22),
                ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 10),
                ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"), 10))   # thin strokes vanish at 1 bit
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


BIG, BOLD, SMALL = fonts()


def text_size(d, txt, font):
    """Pillow 5.1 on the rover has textsize(); Pillow 10 (previews on a Mac) only textbbox()."""
    if hasattr(d, "textbbox"):
        l, t, r, b = d.textbbox((0, 0), txt, font=font)
        return r - l, b - t
    return d.textsize(txt, font=font)


def splash():
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    tw, th = text_size(d, "DHWANI", BIG)
    d.text(((W - tw) // 2, (H - th) // 2 - 3), "DHWANI", font=BIG, fill=1)
    return img


def screen(v, ip, ap, st, line3):
    """v: smoothed pack volts or None; st: /executive/state dict (may be empty)."""
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.text((0, -1), "DHWANI", font=BOLD, fill=1)
    # Wi-Fi: filled dot = joined to a network, ring = only the rover's own hotspot
    if ip:
        d.ellipse((52, 3, 57, 8), fill=1)
    elif ap:
        d.ellipse((52, 3, 57, 8), outline=1)
    # battery icon + charge left
    pct = pack_pct(v) if v is not None else None
    d.rectangle((66, 2, 84, 9), outline=1)
    d.rectangle((85, 4, 86, 7), fill=1)
    if pct is not None and pct > 0:
        d.rectangle((68, 4, 68 + int(round(14 * pct / 100.0)), 7), fill=1)
    txt = ("%d%%" % round(pct)) if pct is not None else "--%"
    tw, _ = text_size(d, txt, BOLD)
    d.text((W - tw - 2, -1), txt, font=BOLD, fill=1)
    # line 2: volts and distance driven
    volts = ("%.2f V" % v) if v is not None else "-- V"
    odo = st.get("odo_m")
    dist = ("%.1f m" % odo) if odo is not None else "-- m"
    d.text((0, 10), volts, font=SMALL, fill=1)
    tw, _ = text_size(d, dist, SMALL)
    d.text((W - tw - 2, 10), dist, font=SMALL, fill=1)
    d.text((0, 21), line3, font=SMALL, fill=1)
    return img


def line3_options(ip, ap, st, fresh):
    out = [("IP " + ip) if ip else ("Hotspot 10.42.0.1" if ap else "No Wi-Fi")]
    if not fresh:
        out.append("AI starting..." if not st else "AI not running")
        return out
    phase = st.get("phase") or "?"
    task = (st.get("mission") or {}).get("task")
    words = (task or "").replace("_", " ").lower()
    if st.get("armed"):
        out.append("Auto: " + words)                  # "DRIVING" merges into "DRIVNG" at 10 px
    elif task and phase not in ("IDLE", "DONE"):
        out.append("%s: %s" % (phase.capitalize(), words))
    else:
        out.append("Mission " + phase.lower())
    ppl = [p for p in (st.get("persons") or []) if p.get("confirmed")]
    out.append("People found %d" % len(ppl))
    return out


class RosState(object):
    """/executive/state in a background thread; the display runs whether or not ROS is up."""

    def __init__(self):
        self.st, self.t = {}, 0.0
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            import rospy
            from std_msgs.msg import String
        except Exception as e:
            log("no rospy (%s): battery and Wi-Fi only" % e)
            return

        def cb(m):
            try:
                self.st = json.loads(m.data)
                self.t = time.time()
            except ValueError:
                pass
        rospy.init_node("dhwani_oled", anonymous=True, disable_signals=True)
        rospy.Subscriber("/executive/state", String, cb, queue_size=1)
        rospy.spin()


def run():
    t0 = time.time()
    oled, v, v_at, net, net_at = None, None, 0.0, (None, False), 0.0
    ros = RosState()
    while True:
        now = time.time()
        try:
            if now - v_at >= BATTERY_READ_S or now < v_at:
                v_at = now
                try:
                    x = pack_volts()
                    v = x if v is None else v + min(1.0, BATTERY_READ_S / BATTERY_TAU_S) * (x - v)
                except Exception:
                    pass
            if now - net_at >= 5.0 or now < net_at:
                net, net_at = wifi(), now
            if oled is None:
                oled = Oled()
                log("display on")
            if now - t0 < SPLASH_S:
                img = splash()
            else:
                fresh = now - ros.t < STATE_STALE_S
                opts = line3_options(net[0], net[1], ros.st, fresh)
                img = screen(v, net[0], net[1], ros.st if fresh else {}, opts[int(now / ROTATE_S) % len(opts)])
            oled.show(img)
        except OSError as e:                      # I2C glitch: reopen next second
            log("I2C error (%s), retrying" % e)
            oled = None
        time.sleep(1.0)


def preview(folder):
    os.makedirs(folder, exist_ok=True)
    st = {"odo_m": 12.3, "phase": "ACTIVE", "armed": True, "mission": {"task": "MAP_AREA"},
          "persons": [{"confirmed": True}, {"confirmed": True}, {"confirmed": False}]}
    shots = {"1_splash": splash(),
             "2_wifi_ip": screen(12.01, "10.253.220.44", True, st, "IP 10.253.220.44"),
             "3_driving": screen(11.59, "10.253.220.44", True, st, line3_options("10.253.220.44", True, st, True)[1]),
             "4_hotspot_only": screen(10.9, None, True, {}, "Hotspot 10.42.0.1")}
    for name, img in shots.items():
        img.convert("L").resize((W * 6, H * 6), Image.NEAREST).save(os.path.join(folder, name + ".png"))
    print("wrote", ", ".join(sorted(shots)))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        preview(sys.argv[2])
    else:
        run()
