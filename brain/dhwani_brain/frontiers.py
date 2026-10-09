# -*- coding: utf-8 -*-
"""Frontiers: known-free cells that touch unknown space, grouped into goals.

A frontier cell is free (not an obstacle, not in a blocked region's margin) and 4-adjacent to an
unknown cell. Cells are clustered by 8-connectivity with a small union-find (frontiers are a few
hundred cells, so plain Python is cheaper than any vectorised labelling here). A cluster longer
than frontier_split_m (a wide chamber mouth, an open stope) is cut along its principal axis so
each piece is a sensible single goal.

Each cluster's GOAL is the reachable cell nearest its centroid: the centroid itself often lies in
unknown space or in a wall's inflation band, and move_base must never be handed such a point.
Clusters with no reachable cell within goal_snap_m are dropped: the opening is too narrow for the
robot, or it sits behind a blocked region.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

from .mapview import dilate4


def frontier_mask(view):
    return view.free & ~view.near_blocked & dilate4(view.unknown)


def _clusters(mask):
    ii, jj = np.nonzero(mask)
    n = int(ii.size)
    if n == 0:
        return ii, jj, []
    h, w = mask.shape
    idx = np.full((h + 2, w + 2), -1, dtype=np.int32)
    idx[ii + 1, jj + 1] = np.arange(n, dtype=np.int32)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):     # half the 8-neighbourhood suffices
        nb = idx[ii + 1 + di, jj + 1 + dj]
        src = np.flatnonzero(nb >= 0)
        for a, b in zip(src.tolist(), nb[src].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(a) for a in range(n)], dtype=np.int64)
    order = np.argsort(roots, kind="mergesort")
    r_sorted = roots[order]
    cuts = np.flatnonzero(np.diff(r_sorted)) + 1
    groups = np.split(order, cuts)
    return ii, jj, groups


def _split(ii, jj, members, max_cells):
    """Cut a long cluster into chunks of at most max_cells along its principal axis."""
    y = ii[members].astype(np.float64)
    x = jj[members].astype(np.float64)
    if members.size < 2:
        return [members]
    y0, x0 = y - y.mean(), x - x.mean()
    cov = np.array([[np.dot(x0, x0), np.dot(x0, y0)], [np.dot(x0, y0), np.dot(y0, y0)]])
    evals, evecs = np.linalg.eigh(cov)
    ax = evecs[:, int(np.argmax(evals))]
    t = x0 * ax[0] + y0 * ax[1]
    span = float(t.max() - t.min())
    k = int(math.ceil((span + 1.0) / max(max_cells, 1.0)))
    if k <= 1:
        return [members]
    edges = np.linspace(t.min(), t.max() + 1e-6, k + 1)
    bins = np.searchsorted(edges, t, side="right") - 1
    return [members[bins == b] for b in range(k) if np.any(bins == b)]


class Frontier(object):
    __slots__ = ("cells_i", "cells_j", "cx", "cy", "size_m", "goal_cell", "gx", "gy", "gyaw",
                 "node", "d_robot", "d_home", "d_link", "unknown_frac", "openness")

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


def find_frontiers(view, reach_img, robot_xy, snap_m=1.5, min_cells=3, split_m=3.0):
    """Frontier clusters with a reachable goal each.

    reach_img: H x W float, distance from the robot (inf = unreachable) - used to snap goals.
    Returns a list of Frontier objects (goal on a reachable cell; yaw facing the frontier).
    """
    mask = frontier_mask(view)
    ii, jj, groups = _clusters(mask)
    out = []
    if not groups:
        return out, mask
    cell = view.cell
    max_cells = split_m / cell
    R = int(math.ceil(snap_m / cell))
    H, W = view.shape
    finite = np.isfinite(reach_img)
    for g in groups:
        if g.size < min_cells:
            continue
        for m in _split(ii, jj, g, max_cells):
            if m.size < max(2, min_cells // 2):
                continue
            ci = ii[m]
            cj = jj[m]
            cyi = float(ci.mean())
            cxj = float(cj.mean())
            a0, a1 = max(0, int(cyi) - R), min(H, int(cyi) + R + 2)
            b0, b1 = max(0, int(cxj) - R), min(W, int(cxj) + R + 2)
            wi, wj = np.nonzero(finite[a0:a1, b0:b1])
            if wi.size == 0:
                continue
            wi = wi + a0
            wj = wj + b0
            d2 = (wi - cyi) ** 2 + (wj - cxj) ** 2
            # nearest to the centroid; small tie-break toward the robot so goals sit on the
            # near lip of the opening rather than behind it
            score = d2 + 0.05 * reach_img[wi, wj]
            k = int(np.argmin(score))
            if d2[k] > (R + 0.5) ** 2:
                continue
            fr = Frontier()
            fr.cells_i, fr.cells_j = ci, cj
            fr.cx = float(view.x0 + (cxj + 0.5) * cell)
            fr.cy = float(view.y0 + (cyi + 0.5) * cell)
            fr.size_m = float(m.size * cell)
            fr.goal_cell = (int(wi[k]), int(wj[k]))
            gx, gy = view.xy_of(wi[k], wj[k])
            fr.gx, fr.gy = float(gx), float(gy)
            dx, dy = fr.cx - fr.gx, fr.cy - fr.gy
            if dx * dx + dy * dy < 0.09:          # goal is on the frontier: face away from robot
                dx, dy = fr.gx - robot_xy[0], fr.gy - robot_xy[1]
            fr.gyaw = math.atan2(dy, dx)
            out.append(fr)
    return out, mask


def _block_clusters(mask, b):
    """8-connected clusters of b x b blocks that contain any set cell. Returns a label image
    (per cell, -1 = none) and the number of clusters."""
    h, w = mask.shape
    hb, wb = (h + b - 1) // b, (w + b - 1) // b
    pad = np.zeros((hb * b, wb * b), dtype=bool)
    pad[:h, :w] = mask
    blk = pad.reshape(hb, b, wb, b).any(axis=3).any(axis=1)
    bi, bj = np.nonzero(blk)
    n = int(bi.size)
    lab_img = np.full((h, w), -1, dtype=np.int32)
    if n == 0:
        return lab_img, 0
    idx = np.full((hb + 2, wb + 2), -1, dtype=np.int32)
    idx[bi + 1, bj + 1] = np.arange(n, dtype=np.int32)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
        nb = idx[bi + 1 + di, bj + 1 + dj]
        src = np.flatnonzero(nb >= 0)
        for a, c in zip(src.tolist(), nb[src].tolist()):
            ra, rc = find(a), find(c)
            if ra != rc:
                parent[max(ra, rc)] = min(ra, rc)
    roots = np.array([find(a) for a in range(n)], dtype=np.int64)
    uniq, lab = np.unique(roots, return_inverse=True)
    bl = np.full((hb, wb), -1, dtype=np.int32)
    bl[bi, bj] = lab
    full = np.repeat(np.repeat(bl, b, axis=0), b, axis=1)[:h, :w]
    lab_img = np.where(mask, full, -1).astype(np.int32)
    return lab_img, int(uniq.size)


class SearchGoal(object):
    __slots__ = ("gx", "gy", "gyaw", "goal_cell", "area_m2", "cx", "cy")


def search_goals(view, reach_img, covered_img, fmask, robot_xy, min_area_m2=2.0, block_m=1.0):
    """Known, reachable floor the camera has not been near yet (covered_img False), grouped into
    ~1 m blocks and clustered. One goal per cluster: its reachable cell nearest the centroid.
    Areas that touch a lidar frontier are left to the frontier."""
    cell = view.cell
    unc = np.isfinite(reach_img) & ~covered_img
    if not unc.any():
        return []
    b = max(1, int(round(block_m / cell)))
    lab, n = _block_clusters(unc, b)
    if n == 0:
        return []
    near_front = dilate4(dilate4(dilate4(fmask)))
    bad = np.unique(lab[near_front & (lab >= 0)])
    ii, jj = np.nonzero(lab >= 0)
    ll = lab[ii, jj]
    counts = np.bincount(ll, minlength=n)
    sy = np.bincount(ll, weights=ii, minlength=n)
    sx = np.bincount(ll, weights=jj, minlength=n)
    min_cells = min_area_m2 / (cell * cell)
    out = []
    badset = set(bad.tolist())
    order = np.argsort(ll, kind="mergesort")
    cuts = np.flatnonzero(np.diff(ll[order])) + 1
    for grp in np.split(order, cuts):
        if grp.size == 0:
            continue
        L = int(ll[grp[0]])
        if counts[L] < min_cells or L in badset:
            continue
        cyi, cxj = sy[L] / counts[L], sx[L] / counts[L]
        gi, gj = ii[grp], jj[grp]
        k = int(np.argmin((gi - cyi) ** 2 + (gj - cxj) ** 2))
        sg = SearchGoal()
        sg.goal_cell = (int(gi[k]), int(gj[k]))
        gx, gy = view.xy_of(gi[k], gj[k])
        sg.gx, sg.gy = float(gx), float(gy)
        sg.cx = float(view.x0 + (cxj + 0.5) * cell)
        sg.cy = float(view.y0 + (cyi + 0.5) * cell)
        sg.area_m2 = float(counts[L] * cell * cell)
        sg.gyaw = math.atan2(sg.gy - robot_xy[1], sg.gx - robot_xy[0])
        out.append(sg)
    return out
