#!/usr/bin/env python
"""OAK-D: cliff/obstacle scan AND spatial person detection, from one pipeline.

ONE NODE ON PURPOSE. Only one process can hold the OAK device, so depth and detection cannot be
separate nodes. They also want the same stereo output anyway - the spatial detection network
fuses the detector with the depth map ON the Myriad X, which is the entire reason this camera is
worth having. The Jetson receives (class, confidence, x, y, z in metres); it runs no inference,
so none of the CUDA 10.2 / Python 3.6 walls apply.

WHAT THE LIDAR CANNOT SEE: the RPLidar sweeps one horizontal plane. A descending staircase
returns nothing in that plane, so to move_base it is indistinguishable from open floor. That is
why the rover drove at the stairs on 2026-09-23. This node reports the two things that plane
misses - floor that is LOWER than it should be (a drop), and obstacles above or below the beam -
and encodes both as ranges on /oak/scan so the existing 2D costmap avoids them with no planner
changes and no new costmap layer.

FAILING LOUD, which matters more than either feature. This hardware is intermittent: on
2026-09-23 it delivered frames at the correct rate and resolution whose every pixel was the same
constant, because the sensors had no power (the board needs its 5 V barrel jack; USB alone is
under the 2.5-3 W the datasheet requires just to stream). A depth node that dies quietly is
WORSE than no node, because the costmap then believes the world is clear precisely where it was
told to trust it. So: no frames, or frames with zero variance, and this cancels the goal, zeroes
/cmd_vel and shouts on /oak/status. It never just stops publishing.

PYTHON 3.6, NOT 2.7, and that forces one design choice. depthai publishes no cp27 wheel at all,
so this cannot run under Melodic's default interpreter. rospy itself is pure Python and imports
happily under 3.6 (verified: publish succeeds, subscribers connect), but `tf` is a C extension
built for 2.7 and raises ImportError. So this node does NO frame transforms - it publishes
detections in base_footprint and mission_node.py, which runs under 2.7 and has tf, converts them
to the map frame. Keeping the transform on the 2.7 side is also where it belongs: that node
already owns the map.

Install:  python3 -m pip install 'depthai==2.26.0.0'   (last line with a cp36 aarch64 wheel)
Run with: PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages python3 oak_node.py
"""
import math
import os

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String, Bool
from actionlib_msgs.msg import GoalID

import depthai as dai

# --- mounting: MEASURE THESE with a ruler. Cliff geometry is entirely determined by them. -----
CAM_HEIGHT  = rospy.get_param("~cam_height", 0.16)    # m, optical centre above the floor
CAM_PITCH   = rospy.get_param("~cam_pitch", 0.35)     # rad, downward tilt
CAM_FORWARD = rospy.get_param("~cam_forward", 0.10)   # m, ahead of base_footprint

# --- what counts as dangerous ------------------------------------------------------------------
CLIFF_DROP  = 0.06     # m below the floor plane before it is a cliff
OBST_MIN_H  = 0.05     # m above floor to be an obstacle (ignores floor noise)
OBST_MAX_H  = 0.60     # above this we fit underneath
MAX_RANGE   = 4.0
MIN_RANGE   = 0.20
FOV         = math.radians(70.0)
NBINS       = 121
FRAME_ID    = "oak_depth_frame"

WATCHDOG_S  = 1.5
MIN_STD     = 1.0      # a depth frame flatter than this is the constant-fill failure

BLOB = os.path.expanduser(rospy.get_param("~blob", "~/models/mobilenet-ssd_openvino_2021.4_6shave.blob"))
# MobileNet-SSD VOC class order; index 15 is person
LABELS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
          "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor"]
PERSON = 15
CONF_MIN = 0.6


class Oak(object):
    def __init__(self):
        self.scan_pub = rospy.Publisher("/oak/scan", LaserScan, queue_size=1)
        self.cliff_pub = rospy.Publisher("/oak/cliff", Bool, queue_size=1)
        self.person_pub = rospy.Publisher("/oak/person", PointStamped, queue_size=10)
        self.status_pub = rospy.Publisher("/oak/status", String, queue_size=1, latch=True)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self.last_good = rospy.Time.now()
        self.halted = False
        self.have_nn = os.path.exists(BLOB)
        if not self.have_nn:
            rospy.logwarn("no blob at %s - depth only, no person detection", BLOB)

    # ---- safety ---------------------------------------------------------------------------
    def halt(self, why):
        if not self.halted:
            rospy.logerr("OAK: %s - HALTING", why)
            self.status_pub.publish(String("HALT: %s" % why))
            self.halted = True
        self.cancel_pub.publish(GoalID())
        self.cmd_pub.publish(Twist())

    def healthy(self):
        if self.halted:
            rospy.loginfo("OAK: frames recovered")
            self.status_pub.publish(String("OK"))
            self.halted = False

    # ---- depth -> scan --------------------------------------------------------------------
    def to_scan(self, depth_m, fx, fy, cx, cy):
        h, w = depth_m.shape
        step = 4
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        z = depth_m[::step, ::step]
        valid = (z > MIN_RANGE) & (z < MAX_RANGE)

        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
        bx = z * cp + Y * sp + CAM_FORWARD      # forward
        by = -X                                 # left
        bz = -Y * cp + z * sp + CAM_HEIGHT      # up, floor at 0

        ranges = np.full(NBINS, float("inf"))
        is_cliff = valid & (bz < -CLIFF_DROP)
        is_obst = valid & (bz > OBST_MIN_H) & (bz < OBST_MAX_H)
        hit = is_cliff | is_obst
        if not hit.any():
            return ranges, False
        hx, hy = bx[hit], by[hit]
        bearing = np.arctan2(hy, hx)
        dist = np.hypot(hx, hy)
        band = np.abs(bearing) <= FOV / 2.0
        if band.any():
            idx = np.clip(((bearing[band] + FOV / 2.0) / FOV * (NBINS - 1)).astype(int), 0, NBINS - 1)
            np.minimum.at(ranges, idx, dist[band])
        return ranges, bool(is_cliff.any())

    def publish_scan(self, ranges):
        m = LaserScan()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = FRAME_ID
        m.angle_min = -FOV / 2.0
        m.angle_max = FOV / 2.0
        m.angle_increment = FOV / (NBINS - 1)
        m.range_min = MIN_RANGE
        m.range_max = MAX_RANGE
        m.ranges = [float(r) if np.isfinite(r) else float("inf") for r in ranges]
        self.scan_pub.publish(m)

    # ---- detections -> map points ----------------------------------------------------------
    def publish_person(self, det):
        """Spatial coords arrive in mm in the CAMERA frame; publish in base_footprint.

        Deliberately NOT transformed to map here - tf has no Python 3 build on Melodic. The
        mission node runs under 2.7, has tf, and owns the map anyway, so it does the conversion.
        """
        p = PointStamped()
        p.header.frame_id = "base_footprint"
        p.header.stamp = rospy.Time.now()
        zc = det.spatialCoordinates.z / 1000.0
        xc = det.spatialCoordinates.x / 1000.0
        yc = det.spatialCoordinates.y / 1000.0
        cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
        p.point.x = zc * cp + yc * sp + CAM_FORWARD
        p.point.y = -xc
        p.point.z = -yc * cp + zc * sp + CAM_HEIGHT
        self.person_pub.publish(p)

    # ---- pipeline ---------------------------------------------------------------------------
    def build(self):
        p = dai.Pipeline()
        ml = p.create(dai.node.MonoCamera); ml.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mr = p.create(dai.node.MonoCamera); mr.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for m in (ml, mr):
            m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            m.setFps(10)
        st = p.create(dai.node.StereoDepth)
        st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        st.setLeftRightCheck(True)
        ml.out.link(st.left); mr.out.link(st.right)

        xd = p.create(dai.node.XLinkOut); xd.setStreamName("depth"); st.depth.link(xd.input)

        if self.have_nn:
            cam = p.create(dai.node.ColorCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setPreviewSize(300, 300)      # MobileNet-SSD input size
            cam.setInterleaved(False)
            cam.setFps(10)
            st.setDepthAlign(dai.CameraBoardSocket.RGB)
            nn = p.create(dai.node.MobileNetSpatialDetectionNetwork)
            nn.setBlobPath(BLOB)
            nn.setConfidenceThreshold(CONF_MIN)
            nn.setBoundingBoxScaleFactor(0.5)
            nn.setDepthLowerThreshold(200)
            nn.setDepthUpperThreshold(8000)
            cam.preview.link(nn.input)
            st.depth.link(nn.inputDepth)
            xn = p.create(dai.node.XLinkOut); xn.setStreamName("det")
            nn.out.link(xn.input)
        return p

    def run(self):
        with dai.Device(self.build()) as dev:
            rospy.loginfo("OAK connected, USB %s, detection=%s",
                          dev.getUsbSpeed(), "on" if self.have_nn else "OFF (no blob)")
            self.status_pub.publish(String("connected USB %s" % dev.getUsbSpeed()))
            cal = dev.readCalibration()
            K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 640, 400))
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]

            qd = dev.getOutputQueue("depth", 4, False)
            qn = dev.getOutputQueue("det", 4, False) if self.have_nn else None
            rate = rospy.Rate(20)
            while not rospy.is_shutdown():
                pkt = qd.tryGet()
                if pkt is not None:
                    d = pkt.getFrame()
                    if float(np.std(d)) < MIN_STD:
                        self.halt("depth frame has zero variance (sensors not reading out - "
                                  "check the 5 V barrel jack)")
                    else:
                        self.last_good = rospy.Time.now()
                        self.healthy()
                        ranges, cliff = self.to_scan(d.astype(np.float32) / 1000.0, fx, fy, cx, cy)
                        self.publish_scan(ranges)
                        self.cliff_pub.publish(Bool(cliff))
                        if cliff:
                            rospy.logwarn_throttle(2.0, "OAK: CLIFF ahead")
                if qn is not None:
                    n = qn.tryGet()
                    if n is not None:
                        for det in n.detections:
                            if det.label == PERSON and det.confidence >= CONF_MIN:
                                self.publish_person(det)
                if (rospy.Time.now() - self.last_good).to_sec() > WATCHDOG_S:
                    self.halt("no usable depth for %.1f s" % WATCHDOG_S)
                rate.sleep()


if __name__ == "__main__":
    rospy.init_node("oak")
    try:
        Oak().run()
    except Exception as e:
        rospy.logfatal("OAK node died: %s", e)
        cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        cancel = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        rospy.sleep(0.5)
        cancel.publish(GoalID())
        for _ in range(20):
            cmd.publish(Twist()); rospy.sleep(0.05)
        raise
