#!/usr/bin/env python3
"""Base station for the rover's LoRa dead-man's switch. Runs on the laptop.

Holds the rover's permission to move. While this is running and you have not pressed stop, it
sends "GO" once a second; the rover halts on its own if those stop arriving for 5 seconds. So
closing this window, unplugging the radio, or walking out of range all stop the robot - which
is the behaviour we wanted on 2026-09-23 and did not have.

Keys:  s = STOP now and hold      g = resume sending GO      q = quit (rover halts in 5 s)

Deliberately no curses, no dependencies beyond pyserial: this is the thing you reach for when
the robot is doing something wrong, so it must start instantly and never crash.
"""
import argparse
import sys
import time
import threading
import serial
import serial.tools.list_ports


def pick_port(explicit):
    if explicit:
        return explicit
    cands = [p.device for p in serial.tools.list_ports.comports()
             if any(k in (p.device or "") for k in ("usbserial", "usbmodem", "ttyUSB", "ttyACM"))]
    if not cands:
        sys.exit("no USB serial port found - pass --port")
    if len(cands) > 1:
        print("multiple ports: %s  (using the first; pass --port to choose)" % cands)
    return cands[0]


class Base:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0)
        self.permit = True
        self.seq = 0
        self.running = True
        self.last_rx = None
        self.last_msg = "(nothing yet)"
        self.rx_count = 0
        self.buf = b""

    def tx_loop(self):
        while self.running:
            self.seq += 1
            cmd = "GO" if self.permit else "STOP"
            try:
                self.ser.write(("B,%d,%s\n" % (self.seq, cmd)).encode())
            except Exception as e:
                print("\nTX failed: %s" % e)
            time.sleep(1.0)

    def rx_loop(self):
        while self.running:
            try:
                n = self.ser.in_waiting
                if n:
                    self.buf += self.ser.read(n)
                while b"\n" in self.buf:
                    line, self.buf = self.buf.split(b"\n", 1)
                    s = line.strip().decode("ascii", "ignore")
                    if s.startswith("R,"):
                        self.rx_count += 1
                        self.last_rx = time.time()
                        self.last_msg = s
            except Exception:
                pass
            time.sleep(0.05)

    def status_loop(self):
        while self.running:
            age = ("%.1fs" % (time.time() - self.last_rx)) if self.last_rx else "never"
            state = "GO  " if self.permit else "STOP"
            sys.stdout.write("\r[%s] rx=%4d  last=%6s  %-46s" %
                             (state, self.rx_count, age, self.last_msg))
            sys.stdout.flush()
            time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=9600)
    a = ap.parse_args()
    port = pick_port(a.port)
    print("base station on %s @ %d" % (port, a.baud))
    print("  s = STOP      g = resume GO      q = quit (rover halts in 5 s)\n")
    b = Base(port, a.baud)
    for fn in (b.tx_loop, b.rx_loop, b.status_loop):
        threading.Thread(target=fn, daemon=True).start()
    try:
        while True:
            k = sys.stdin.readline().strip().lower()
            if k == "s":
                b.permit = False; print("\n>>> STOP sent and held")
            elif k == "g":
                b.permit = True;  print("\n>>> GO resumed")
            elif k == "q":
                break
    except KeyboardInterrupt:
        pass
    b.running = False
    print("\nbase station stopped - rover will halt within 5 s")


if __name__ == "__main__":
    main()
