import numpy as np
import tensorflow as tf

from core import box_list, box_list_ops


def pruning_overlap_box(boxes, scores):
    corners = tf.constant(np.squeeze(boxes), tf.float32)
    boxes_list = box_list.BoxList(corners)
    boxes_list.add_field('scores', tf.constant(np.squeeze(scores)))
    iou_thresh = 0.1
    max_output_size = 100
    sess = tf.compat.v1.Session()
    nms = box_list_ops.non_max_suppression(
        boxes_list, iou_thresh, max_output_size)
    boxes = sess.run(nms.get())
    return np.array(boxes), np.array(scores)


def pruning_overlap_box_on_score(
        boxes,
        scores,
        min_score_thresh=0.6):
    '''
    :param boxes: Shape (N, 4)
    :param scores: shape(N)
    :param min_score_thresh: 0.6
    :return: boxes subset which is pruning by score
    '''
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)

    box_passed_thresh = [boxes[i] for i in range(boxes.shape[0]) if scores[i] > min_score_thresh]
    score_passed_thresh = [scores[i] for i in range(boxes.shape[0]) if scores[i] > min_score_thresh]

    if len(box_passed_thresh) <= 1:
        return np.array(box_passed_thresh), np.array(score_passed_thresh)

    return pruning_overlap_box(
        box_passed_thresh,
        score_passed_thresh)