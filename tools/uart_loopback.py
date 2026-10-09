#!/usr/bin/env python
"""Prove which two holes on the header are UART TX and RX, with one jumper wire.

Photographs of a 40-pin header are not good enough to tell which physical row a wire is in,
and guessing wrong costs a round trip every time. Bridging TX to RX makes the hardware answer
for itself: send bytes, and if the same bytes come back, the jumper is across pins 8 and 10.
Nothing else on the header produces an echo, so a pass is unambiguous.
"""
import sys, time, serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyTHS1"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 9600

PROBE = b"LOOPBACK-TEST-1234567890\n"

try:
    s = serial.Serial(PORT, BAUD, timeout=0.5)
except Exception as e:
    print("cannot open %s: %s" % (PORT, e))
    sys.exit(2)

s.reset_input_buffer()
s.reset_output_buffer()
s.write(PROBE)
s.flush()
time.sleep(0.8)

n = s.in_waiting
got = s.read(n) if n else b""
s.close()

print("sent    : %r" % PROBE)
print("received: %r" % got)
print()
if got.strip() == PROBE.strip():
    print("PASS - perfect echo.")
    print("Those two holes ARE pins 8 and 10. Remove the jumper and put white in one,")
    print("green in the other. If the module then stays silent, swap just those two.")
    sys.exit(0)
elif got:
    print("PARTIAL - something came back but it does not match.")
    print("Usually a loose jumper or a baud mismatch. Reseat both ends and retry.")
    sys.exit(1)
else:
    print("FAIL - nothing came back.")
    print("Move BOTH ends of the jumper to the other row, same two columns, and retry.")
    sys.exit(1)
