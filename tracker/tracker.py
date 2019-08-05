from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag

from myutils.const import Const


class Tracker():
    @staticmethod
    def check_point_is_center_box(cen_y, cen_x, y_min, x_min, y_max, x_max):
        is_valid_x = (x_min < cen_x < x_max)
        is_valid_y = (y_min < cen_y < y_max)
        return is_valid_x and is_valid_y

    @staticmethod
    def get_center(box):
        y_min, x_min, y_max, x_max = box
        return (y_min + y_max) / 2, (x_min + x_max) / 2

    @staticmethod
    def is_center_door(cen_x, cen_y):
        y_door_min, x_door_min = Const.get_start_ratio()
        y_door_max, x_door_max = Const.get_end_ratio()
        return Tracker.check_point_is_center_box(cen_y, cen_x, y_door_min, x_door_min, y_door_max, x_door_max)

    @staticmethod
    def assign_to_tracker(boxes,
                          TRACKER_BOX_MAP,
                          track_ids,
                          num_come_in,
                          num_come_out):
        if len(TRACKER_BOX_MAP) == 0:
            for i in range(len(boxes)):
                box = tuple(boxes[i])
                y_min, x_min, y_max, x_max = box
                cen_y, cen_x = Tracker.get_center(box)
                TRACKER_BOX_MAP[i].append((y_min, x_min, y_max, x_max))
                track_ids.append(i)
                if Tracker.is_center_door(cen_y, cen_x):
                    num_come_in += 1
        else:
            for box in boxes:
                cen_y, cen_x = Tracker.get_center(box)

                # Determine box is in or not in TRACKER_BOX_MAP
                id_box_tracked = None
                for track_id, track_box in TRACKER_BOX_MAP.items():
                    track_y_min, track_x_min, track_y_max, track_x_max = np.squeeze(track_box)
                    is_between = Tracker.check_point_is_center_box(
                        cen_y,
                        cen_x,
                        track_y_min,
                        track_x_min,
                        track_y_max,
                        track_x_max)
                    if is_between:
                        id_box_tracked = track_id
                        break

                if id_box_tracked is not None:
                    TRACKER_BOX_MAP[id_box_tracked] = box
                else:
                    new_id = track_ids[-1] + 1
                    track_ids.append(new_id)
                    TRACKER_BOX_MAP[new_id].append(box)

                    # If tracker is in center door then add to the come in
                    if Tracker.is_center_door(cen_y, cen_x):
                        num_come_in += 1

        track_boxes = []
        for track_id, track_box in TRACKER_BOX_MAP.items():
            track_boxes.append(np.squeeze(track_box))

        return TRACKER_BOX_MAP, track_boxes, track_ids, num_come_in, num_come_out
