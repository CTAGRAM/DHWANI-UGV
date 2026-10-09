"""Offline test of gateway.py's fan-out (28 Sep 2026): a frozen viewer must not hold up the others.
Runs on the Mac with ROS stubbed out:  python3 tests/test_send_fanout.py"""
import asyncio, json, os, sys, time, types

HERE = os.path.dirname(os.path.abspath(__file__))
for name in ["rospy", "std_msgs", "std_msgs.msg", "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg",
             "sensor_msgs", "sensor_msgs.msg", "tf2_msgs", "tf2_msgs.msg", "actionlib_msgs", "actionlib_msgs.msg"]:
    m = types.ModuleType(name)
    for attr in ["String", "Bool", "Float32", "Twist", "PoseArray", "OccupancyGrid", "Odometry", "Path",
                 "LaserScan", "Image", "TFMessage", "GoalID"]:
        setattr(m, attr, object)
    sys.modules[name] = m
sys.path.insert(0, os.path.dirname(HERE))
import gateway as G


class Transport(object):
    def __init__(self, buf):
        self.buf = buf

    def get_write_buffer_size(self):
        return self.buf


class WS(object):
    def __init__(self, kind):
        self.kind, self.got = kind, []
        self.transport = Transport(1024 * 1024 if kind == "behind" else 0)

    async def send(self, s):
        if self.kind == "frozen":
            await asyncio.sleep(3600)          # a phone tab asleep on a hotspot
        self.got.append(s)


class C(object):
    def __init__(self, kind):
        self.ws, self.camera, self.depth = WS(kind), True, False


async def main():
    gw = object.__new__(G.Gateway)
    ok, frozen, behind = C("ok"), C("frozen"), C("behind")
    gw.clients = {ok, frozen, behind}
    fails = 0

    def check(name, cond, info=""):
        nonlocal fails
        print("%s %-58s %s" % ("PASS" if cond else "FAIL", name, info))
        fails += 0 if cond else 1

    t0 = time.time(); await gw.broadcast({"t": "pose", "x": 1}); dt = time.time() - t0
    check("pose broadcast finishes despite a frozen viewer", dt < G.SEND_WAIT_S + 0.3, "%.2f s" % dt)
    check("healthy viewer got the pose", len(ok.ws.got) == 1)
    check("viewer far behind skipped the pose", len(behind.ws.got) == 0)

    t0 = time.time(); await gw.broadcast({"t": "control", "controller": "x"}); dt = time.time() - t0
    check("control broadcast bounded too", dt < G.SEND_WAIT_S + 0.3, "%.2f s" % dt)
    check("viewer far behind still gets control (critical)", len(behind.ws.got) == 1)

    t0 = time.time(); await gw.send_camera(1, b"\\xff\\xd8jpeg"); dt = time.time() - t0
    check("camera frame fan-out bounded", dt < G.SEND_WAIT_S + 0.3, "%.2f s" % dt)
    check("healthy viewer got the frame", ok.ws.got[-1][:1] == b"J")

    t0 = time.time(); await gw.send_objects([{"label": "chair", "conf": 0.8, "box": [0, 0, 1, 1], "z": 1.2}], 1.0)
    dt = time.time() - t0
    check("object overlay fan-out bounded", dt < G.SEND_WAIT_S + 0.3, "%.2f s" % dt)
    check("healthy viewer got the objects", json.loads(ok.ws.got[-1])["t"] == "dets")

    # 20 broadcasts back to back (map, scan, stats...) must not pile up behind the frozen viewer
    t0 = time.time()
    await asyncio.gather(*[gw.broadcast({"t": "scan", "i": i}) for i in range(20)])
    dt = time.time() - t0
    check("20 concurrent broadcasts finish together", dt < G.SEND_WAIT_S + 0.5, "%.2f s" % dt)
    print("%d failed" % fails)
    return fails

sys.exit(asyncio.run(main()))
