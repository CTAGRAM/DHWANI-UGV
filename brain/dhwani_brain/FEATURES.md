# DHWANI brain features

Generated from `dhwani_brain/features.py` (`python -m dhwani_brain.features`).
Every value is computed on the rover from the `obs` dict alone (plus the brain's own
memory of earlier obs dicts: start heading, first-branch commitment, verify attempts,
previous goal, battery and odometry at the first call).

Distances are geodesic (octile wavefront on the 0.2 m planning grid, obstacles and
blocked regions inflated by 0.15 m) unless marked straight-line. Energy uses the
brain's model: pct per metre = drain_pct_per_m + drain_pct_per_s / speed, times
path_factor, raised to the measured rate once 20 m of odometry exist.

## Per-candidate vector (31 values)

Slots: 6 frontier goals (top by rules utility, feasible only), 2 verify standoff
points, RETURN_HOME, RETURN_TO_LINK, HOLD. Missing slots are masked.

| # | name | meaning and normalisation |
|---|---|---|
| 0 | `is_explore` | 1 if the candidate is an exploration goal (lidar frontier or camera search) |
| 1 | `is_verify` | 1 if it is a standoff point to verify an unconfirmed sighting |
| 2 | `is_home` | 1 if it is RETURN_HOME |
| 3 | `is_link` | 1 if it is RETURN_TO_LINK (last pose where the link worked) |
| 4 | `is_hold` | 1 if it is HOLD (stay where you are) |
| 5 | `d_robot` | geodesic robot->goal, m / 50, clipped to [0, 3] |
| 6 | `d_home` | geodesic goal->home, m / 50, clipped to [0, 3]; 3 if home unreachable |
| 7 | `d_euclid` | straight-line robot->goal, m / 50, clipped to [0, 3] |
| 8 | `detour` | geodesic / max(straight line, 1 m), / 3, clipped to [0, 1] |
| 9 | `sin_bear` | sin of goal bearing relative to the robot heading |
| 10 | `cos_bear` | cos of goal bearing relative to the robot heading |
| 11 | `sin_bear0` | sin of goal bearing relative to the heading at mission start |
| 12 | `cos_bear0` | cos of goal bearing relative to the heading at mission start |
| 13 | `branch_match` | 1 if first_branch is still pending and the goal lies in its sector |
| 14 | `is_search` | 1 if it is a camera-search goal (known floor the camera has not been near) |
| 15 | `size` | frontier length m / 5, or unseen search area m2 / 10; clipped to [0, 2]; else 0 |
| 16 | `unknown_frac` | unknown cells / all cells in a 6 m x 6 m box around the goal |
| 17 | `openness` | passable cells / all cells in a 3.4 m x 3.4 m box around the goal |
| 18 | `bat_margin` | (battery - energy to goal - reserve from goal - floor) / 100, [-1, 1] |
| 19 | `time_margin` | (time limit - time needed incl. return) / limit, [-1, 1]; 1 if no limit |
| 20 | `dist_margin` | (distance limit - odometry needed incl. return) / limit, [-1, 1]; 1 if none |
| 21 | `utility` | log1p(rules frontier utility x 10) / 3; 0 for non-frontiers |
| 22 | `rules_pick` | 1 if the rules policy chose this candidate |
| 23 | `is_prev_goal` | 1 if within 1.5 m of the previous decision's goal |
| 24 | `d_home_delta` | (d_home(goal) - d_home(robot)) / 50, [-2, 2]: >0 means deeper |
| 25 | `beyond_link` | (d_home(goal) - d_home(last link point)) / 50, [-2, 2]; 0 if none |
| 26 | `person_near` | exp(-d / 3 m), d = goal to nearest unconfirmed sighting; 0 if none |
| 27 | `visited` | fraction of trail points within 2 m of the goal |
| 28 | `blocked_near` | exp(-d / 3 m), d = goal to nearest blocked region edge; 0 if none |
| 29 | `rank` | rank by rules utility / K among frontiers (0 = best); 1 for non-frontiers |
| 30 | `hits` | sighting hits / 5, clipped to [0, 1] (VERIFY only) |

## Global vector (45 values)

| # | name | meaning and normalisation |
|---|---|---|
| 0 | `battery` | battery_pct / 100 |
| 1 | `reserve_margin` | (battery - floor - reserve to get home now) / 100, [-1, 1] |
| 2 | `link_ok` | 1 if the dashboard heartbeat is live |
| 3 | `since_contact` | since_contact_s / 600, clipped to [0, 1] |
| 4 | `elapsed` | elapsed_s / 3600, clipped to [0, 1.5] |
| 5 | `time_frac` | elapsed / max_duration, [0, 1.5]; 0 if no limit |
| 6 | `has_time_limit` | 1 if max_duration_min is set |
| 7 | `odo` | odo_m / 1000, clipped to [0, 1.5] |
| 8 | `dist_frac` | odo / max_distance, [0, 1.5]; 0 if no limit |
| 9 | `has_dist_limit` | 1 if max_distance_m is set |
| 10 | `d_home_now` | geodesic robot->home, m / 100, clipped to [0, 2]; 2 if unreachable |
| 11 | `home_reachable` | 1 if a known route home exists |
| 12 | `found` | confirmed persons / 3, clipped to [0, 1] |
| 13 | `unreported` | confirmed but not yet reported / 3, clipped to [0, 1] |
| 14 | `unconfirmed` | open (not written-off) unconfirmed sightings / 3, clipped to [0, 1] |
| 15 | `known_free` | known free area, m2 / 2000, clipped to [0, 1.5] |
| 16 | `cam_covered` | fraction of reachable known floor within cam_cover_m of the trail |
| 17 | `n_search` | camera-search goals / 10, clipped to [0, 1.5] |
| 18 | `n_frontiers` | frontier clusters / 10, clipped to [0, 1.5] |
| 19 | `frontier_len` | total frontier length, m / 20, clipped to [0, 1.5] |
| 20 | `n_blocked` | blocked regions / 3, clipped to [0, 1] |
| 21 | `link_known` | 1 if a last-link pose exists |
| 22 | `d_link` | geodesic robot->last-link point, m / 50, [0, 2]; 0 if none or link ok |
| 23 | `branch_pending` | 1 if first_branch is not ANY and not yet taken |
| 24 | `min_battery` | mission min_battery_pct / 100; 0 if none |
| 25 | `speed_slow` | 1 if mission speed is SLOW |
| 26 | `return_when_done` | mission flag |
| 27 | `report_person` | mission flag |
| 28 | `approach_person` | mission flag |
| 29 | `task_search_survivors` | mission task one-hot |
| 30 | `task_map_area` | mission task one-hot |
| 31 | `task_goto_waypoint` | mission task one-hot |
| 32 | `task_return_home` | mission task one-hot |
| 33 | `task_halt` | mission task one-hot |
| 34 | `opf_report_and_continue` | on_person_found one-hot |
| 35 | `opf_report_and_return` | on_person_found one-hot |
| 36 | `opf_stay` | on_person_found one-hot |
| 37 | `oll_continue_then_return` | on_link_loss one-hot |
| 38 | `oll_return_now` | on_link_loss one-hot |
| 39 | `oll_continue` | on_link_loss one-hot |
| 40 | `oll_hold` | on_link_loss one-hot |
| 41 | `fb_any` | first_branch one-hot |
| 42 | `fb_left` | first_branch one-hot |
| 43 | `fb_right` | first_branch one-hot |
| 44 | `fb_straight` | first_branch one-hot |

The learned network additionally standardises each input with the mean and
standard deviation stored in its .npz (computed on training data only).
