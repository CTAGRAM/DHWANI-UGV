#!/usr/bin/env python
"""Publish map junctions as Foxglove markers. The topology trigger from the report, made visible.

our design report, section 2.6 claims relay placement is driven by map TOPOLOGY
rather than link quality, on the argument that a corridor's signal does not degrade measurably
until after the corner, so any reactive "drop when RSSI falls" rule is late by construction.
2.7 then concedes the detector that would fire the drop does not exist and the shipped code
drops on distance alone. This node is that detector.

PORTED TO cv2 ON PURPOSE: the desktop prototype used skimage, which is not installed for the
rover's python2.7 rospy and is not worth building on a 4 GB Nano. Everything here is numpy plus
cv2 3.2, both already present:

    skeletonize          -> cv2.ximgproc.thinning
    distance_transform   -> cv2.distanceTransform
    label                -> cv2.connectedComponentsWithStats
    convolve             -> cv2.filter2D

NOT MACHINE LEARNING, deliberately. Thin the free space to a one-pixel medial axis and the
corridors become a graph; a junction is where that graph branches. It is a morphological
property of the map, not an inference. Feeding an occupancy grid to a network would mean
serialising it to text and destroying the spatial structure that IS the signal.
"""
import math
import numpy as np
import cv2
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

FREE_MAX      = 40      # occupancy below this counts as free
SPUR_M        = 1.2     # branches shorter than this are wall noise, not corridors
MIN_CLEAR_M   = 0.50    # a gap the robot cannot fit through is not a junction
MERGE_M       = 1.5     # two detections closer than this are one junction seen twice
MIN_CORRIDORS = 3       # a junction has at least three ways out
ERODE_PX      = 1
PERIOD_S      = 10.0    # recompute this often; thinning a 1000x1200 grid is not free


def _neighbour_count(binary):
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
    return cv2.filter2D(binary.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)


def _prune_spurs(skel, iters):
    """Strip endpoints `iters` times; annihilates every branch shorter than that.

    Without this the detector is useless: raw thinning sprouts a spur at every bump in a wall
    and each spur root reads as a branch point. On the 2026-09-23 map that was 127 'junctions'
    in a room with about a dozen real openings. Pruning to 1.2 m leaves 28.
    """
    s = skel.copy()
    for _ in range(iters):
        ends = s & (_neighbour_count(s) == 1)
        if not ends.any():
            break
        s = s & ~ends
    return s


def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == biggest


def find_junctions(grid, res, ox, oy):
    """grid: int8 occupancy (-1 unknown). Returns [(x, y, corridors, clearance_m), ...]."""
    free = (grid >= 0) & (grid < FREE_MAX)
    if ERODE_PX:
        free = cv2.erode(free.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=ERODE_PX) > 0
    free = _largest_component(free)
    if not free.any():
        return []

    skel = cv2.ximgproc.thinning((free * 255).astype(np.uint8)) > 0
    skel = _prune_spurs(skel, int(round(SPUR_M / res)))
    bp = skel & (_neighbour_count(skel) >= 3)
    if not bp.any():
        return []

    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res

    cl_px = 6
    blob = cv2.dilate(bp.astype(np.uint8), np.ones((cl_px, cl_px), np.uint8)) > 0
    n, lab, _, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero((lab == i) & bp)
        if len(xs) == 0:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        arms = _count_arms(skel, lab == i)
        if arms < MIN_CORRIDORS:
            continue
        clear = float(dist[int(round(cy)), int(round(cx))])
        if clear < MIN_CLEAR_M:
            continue
        # OccupancyGrid row 0 is the MIN y edge (unlike a PGM, where row 0 is max y)
        out.append((ox + (cx + 0.5) * res, oy + (cy + 0.5) * res, arms, clear))

    out.sort(key=lambda r: -r[3])
    kept = []
    for j in out:
        if all((j[0] - k[0]) ** 2 + (j[1] - k[1]) ** 2 > MERGE_M ** 2 for k in kept):
            kept.append(j)
    return kept


def _count_arms(skel, cluster):
    ring = (cv2.dilate(cluster.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~cluster & skel
    n, _, _, _ = cv2.connectedComponentsWithStats(ring.astype(np.uint8), 8)
    return max(0, n - 1)


class JunctionNode(object):
    def __init__(self):
        self.mk = rospy.Publisher("/junctions/markers", MarkerArray, queue_size=1, latch=True)
        self.pa = rospy.Publisher("/junctions/poses", PoseArray, queue_size=1, latch=True)
        self.last = 0.0
        self.count = 0
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1)

    def on_map(self, m):
        now = rospy.get_time()
        if now - self.last < PERIOD_S:
            return
        self.last = now
        g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        t0 = rospy.get_time()
        try:
            js = find_junctions(g, m.info.resolution,
                                m.info.origin.position.x, m.info.origin.position.y)
        except Exception as e:
            rospy.logerr("junction detection failed: %s", e)
            return
        dt = rospy.get_time() - t0
        if len(js) != self.count:
            rospy.loginfo("junctions: %d  (%.2f s)", len(js), dt)
            self.count = len(js)
        self.publish(js, m.header.frame_id or "map")

    def publish(self, js, frame):
        arr = MarkerArray()
        # one DELETEALL so stale markers from a previous, larger set do not linger in Foxglove
        d = Marker(); d.action = Marker.DELETEALL; d.header.frame_id = frame
        arr.markers.append(d)

        pa = PoseArray(); pa.header.frame_id = frame; pa.header.stamp = rospy.Time.now()
        for i, (x, y, arms, clear) in enumerate(js):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = rospy.Time.now()
            m.ns = "junctions"; m.id = i
            m.type = Marker.CYLINDER; m.action = Marker.ADD
            m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.02
            m.pose.orientation.w = 1.0
            # radius tracks how open it is, so the best drop sites read biggest on screen
            r = max(0.25, min(1.2, clear * 0.6))
            m.scale.x = r * 2; m.scale.y = r * 2; m.scale.z = 0.04
            top = i < 10
            m.color.r = 0.0; m.color.g = 0.9 if top else 0.5
            m.color.b = 1.0; m.color.a = 0.85 if top else 0.35
            arr.markers.append(m)

            t = Marker()
            t.header = m.header; t.ns = "junction_labels"; t.id = i
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = x; t.pose.position.y = y; t.pose.position.z = 0.35
            t.pose.orientation.w = 1.0
            t.scale.z = 0.22
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 0.9
            t.text = "J%d %dway %.1fm" % (i + 1, arms, clear)
            arr.markers.append(t)

            p = Pose(); p.position.x = x; p.position.y = y; p.orientation.w = 1.0
            pa.poses.append(p)

        self.mk.publish(arr)
        self.pa.publish(pa)


if __name__ == "__main__":
    rospy.init_node("junction_detector")
    JunctionNode()
    rospy.loginfo("junction detector up; publishing /junctions/markers and /junctions/poses")
    rospy.spin()
