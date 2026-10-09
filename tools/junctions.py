"""Find junctions in an occupancy grid by skeletonising the free space.

THE CLAIM THIS IMPLEMENTS: our design report, section 2.6 says relay placement is
triggered by map TOPOLOGY, not by link quality - the argument being that in a corridor the signal
does not degrade measurably until after the corner, so a reactive "drop when RSSI falls" trigger
is always late by construction. 2.7 then admits the belief-side detector that would fire the drop
does not exist, and the shipped code drops on distance alone (FIRST_DROP_M=6.0, DROP_INTERVAL_M=8.0).
This is that missing piece.

HOW IT WORKS, and why it is not machine learning: a corridor's free space, thinned to a
single-pixel-wide skeleton, becomes a graph. Points where that graph branches ARE the junctions -
there is no inference involved, it is a morphological property of the map. A neural network here
would be strictly worse: the input would have to be serialised out of the grid, destroying the
spatial structure that is the entire signal.

  1. free space  = known AND not occupied
  2. erode a little so a doorway does not fuse with the wall beside it
  3. skeletonise  -> one-pixel-wide medial axis
  4. a skeleton pixel with >= 3 skeleton neighbours is a branch point
  5. cluster branch points, because thinning frays a real junction into a rosette of them
  6. score each cluster by how much free space it commands, to rank drop sites

Validated against tour_20260923 - the 313 m2 map the rover actually built on 2026-09-23.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, binary_erosion, disk

FREE_THRESH = 250          # PGM: 254/255 free, 0 occupied, 205 unknown
OCC_THRESH = 100


def free_mask(pgm, erode_px=1):
    """Known-free cells, eroded so narrow gaps do not merge with walls."""
    free = pgm >= FREE_THRESH
    if erode_px > 0:
        free = binary_erosion(free, disk(erode_px))
    # keep only the largest connected component: islands of free space behind a wall are
    # mapping noise, and their skeletons produce phantom junctions
    lab, n = ndimage.label(free)
    if n > 1:
        sizes = ndimage.sum(free, lab, range(1, n + 1))
        free = lab == (int(np.argmax(sizes)) + 1)
    return free


def _neighbours(skel):
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    return ndimage.convolve(skel.astype(int), k, mode="constant")


def prune_spurs(skel, iters):
    """Repeatedly delete endpoints. Removes every branch shorter than `iters` pixels.

    This is the difference between a usable detector and a useless one. Raw skeletonisation of a
    SLAM map sprouts a spur at every bump in a wall, and each spur root registers as a branch
    point - 127 of them on the 313 m2 test map, where the room has perhaps a dozen real openings.
    An endpoint has exactly one skeleton neighbour, so stripping endpoints N times shortens every
    branch by N and annihilates anything shorter than that. Real corridors are metres long and
    survive; 10 cm of wall noise does not.
    """
    s = skel.copy()
    for _ in range(iters):
        ends = s & (_neighbours(s) == 1)
        if not ends.any():
            break
        s = s & ~ends
    return s


def branch_points(skel):
    """Skeleton pixels with 3+ skeleton neighbours."""
    return skel & (_neighbours(skel) >= 3)


def find_junctions(pgm, resolution, origin_xy, erode_px=1, cluster_px=6, min_corridors=3,
                   spur_m=0.8, min_clear_m=0.35, merge_m=1.0):
    """Returns [(x_m, y_m, n_corridors, clearance_m), ...] sorted by clearance, widest first.

    clearance_m is the distance from the junction to the nearest obstacle - a proxy for how
    open the space is. A breadcrumb wants the widest junction it can reach, because that is
    where the radio has line of sight down the most corridors.

    spur_m      branches shorter than this are wall noise, not corridors
    min_clear_m a junction the robot cannot fit through is not a junction
    merge_m     two detections closer than this are one junction seen twice
    """
    free = free_mask(pgm, erode_px)
    skel = skeletonize(free)
    skel = prune_spurs(skel, int(round(spur_m / resolution)))
    bp = branch_points(skel)
    if not bp.any():
        return []

    # distance to nearest non-free cell, in metres
    dist = ndimage.distance_transform_edt(free) * resolution

    # cluster the fraying: dilate the branch mask and label connected blobs
    struct = np.ones((cluster_px, cluster_px), bool)
    lab, n = ndimage.label(ndimage.binary_dilation(bp, struct))
    out = []
    h = pgm.shape[0]
    for i in range(1, n + 1):
        ys, xs = np.nonzero((lab == i) & bp)
        if len(xs) == 0:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        # how many distinct skeleton arms leave this cluster? counts the corridors meeting here
        arms = _count_arms(skel, lab == i)
        if arms < min_corridors:
            continue
        clear = float(dist[int(round(cy)), int(round(cx))])
        if clear < min_clear_m:
            continue
        wx = origin_xy[0] + (cx + 0.5) * resolution
        wy = origin_xy[1] + (h - 1 - cy + 0.5) * resolution   # PGM row 0 is max y
        out.append((wx, wy, arms, clear))

    # one physical junction can still survive as two detections a few cells apart; keep the
    # more open of any pair closer than merge_m
    out.sort(key=lambda r: -r[3])
    kept = []
    for j in out:
        if all((j[0] - k[0]) ** 2 + (j[1] - k[1]) ** 2 > merge_m ** 2 for k in kept):
            kept.append(j)
    return kept


def _count_arms(skel, cluster):
    """Skeleton components remaining after deleting the cluster = corridors meeting there."""
    ring = ndimage.binary_dilation(cluster, np.ones((3, 3), bool)) & ~cluster & skel
    _, n = ndimage.label(ring, structure=np.ones((3, 3), int))
    return int(n)
