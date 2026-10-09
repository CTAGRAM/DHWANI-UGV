#!/usr/bin/env python
"""Ask an E32 for its configuration. The only single-module proof that wiring is correct.

In transparent mode (M0=M1=0) the module never replies to anything, so silence on the port
tells you nothing - a perfectly wired module and a completely disconnected one look identical.
Pulling M0 and M1 HIGH puts it in configuration mode, where C1 C1 C1 returns six bytes. Getting
those six bytes proves three things at once: the module has power, Jetson TX -> module RXD
works, and module TXD -> Jetson RX works.
"""
import sys, time, serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyTHS1"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 9600

AIR = {0: "0.3k", 1: "1.2k", 2: "2.4k", 3: "4.8k", 4: "9.6k", 5: "19.2k", 6: "19.2k", 7: "19.2k"}
UART_BPS = {0: 1200, 1: 2400, 2: 4800, 3: 9600, 4: 19200, 5: 38400, 6: 57600, 7: 115200}
PWR_30D = {0: "30dBm (1W)", 1: "27dBm", 2: "24dBm", 3: "21dBm"}

# Two things silently defeat this probe, both learned the hard way on 24 Sep 2026:
#  1. mission_node.py holds the same port and reads it non-blocking: it eats the six reply bytes
#     before we see them. Pause whoever holds the port (SIGSTOP) for the two seconds this takes.
#  2. M1 is driven low by lora-mode.service (transparent mode). Config mode needs M1 high, so
#     raise header pin 7 (gpio216) for the probe and put it back afterwards. Needs sudo.
import os, signal, subprocess
GPIO = "/sys/class/gpio/gpio216/value"
holders = []
try:
    out = subprocess.check_output(["lsof", "-t", PORT], stderr=subprocess.STDOUT)
    holders = [int(p) for p in out.split() if int(p) != os.getpid()]
except Exception:
    pass
for pid in holders:
    os.kill(pid, signal.SIGSTOP)
if holders:
    print("paused port holder(s) %s for the probe" % holders)
m1_was = None
try:
    m1_was = open(GPIO).read().strip()
    open(GPIO, "w").write("1")
    time.sleep(1.0)
except IOError:
    print("could not drive M1 high via %s (run with sudo) - assuming M0/M1 are already high" % GPIO)

s = serial.Serial(PORT, BAUD, timeout=0.6)
s.reset_input_buffer()
s.write(bytearray([0xC1, 0xC1, 0xC1]))
s.flush()
time.sleep(1.2)

d = s.read(6)
s.close()
if m1_was is not None:
    open(GPIO, "w").write(m1_was)
for pid in holders:
    os.kill(pid, signal.SIGCONT)

if len(d) < 6:
    print("FAILED - got %d bytes, expected 6" % len(d))
    print()
    print("  If you moved M0 and M1 to 3V3 and still get nothing, check in this order:")
    print("   1. swap the white and green wires  (TXD/RXD crossover is the usual cause)")
    print("   2. measure yellow-to-orange with a meter - must read ~3.3 V")
    print("   3. confirm M0 and M1 really are on a 3V3 pin, not left floating")
    sys.exit(1)

b = [ord(c) if isinstance(c, str) else c for c in d]
print("RAW: %s" % " ".join("%02X" % v for v in b))
print()
head, addh, addl, sped, chan, opt = b
print("  header      : 0x%02X %s" % (head, "(saved on power-off)" if head == 0xC0 else "(temporary)"))
print("  address     : 0x%02X%02X" % (addh, addl))
print("  UART baud   : %d 8N1" % UART_BPS.get((sped >> 3) & 0x07, 0))
print("  air rate    : %s bps" % AIR.get(sped & 0x07, "?"))
print("  channel     : %d  (%d MHz)" % (chan & 0x1F, 410 + (chan & 0x1F)))
print("  TX power    : %s" % PWR_30D.get(opt & 0x03, "?"))
print("  transparent : %s" % ("no (fixed)" if opt & 0x80 else "yes"))
print()
print("MODULE IS ALIVE AND CORRECTLY WIRED.")
print("Both modules must match on address, air rate and channel to talk to each other.")
