from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
# Set headless-friendly backend.
import numpy as np

from myutils.rectangles import Rectangle


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)

    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    radius = 10
    x_centroid = (left + right) / 2
    y_centroid = (top + bottom) / 2
    draw.ellipse((x_centroid - radius, y_centroid - radius, x_centroid + radius, y_centroid + radius), fill=(0, 0, 255))
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def is_intersect(box_a, box_b):
    ymin_a, xmin_a, ymax_a, xmax_a = box_a
    ymin_b, xmin_b, ymax_b, xmax_b = box_b
    rect1 = Rectangle(xmin_a, ymin_a, xmax_a, ymax_a)
    rect2 = Rectangle(xmin_b, ymin_b, xmax_b, ymax_b)
    ret = (rect1 & rect2)
    if ret is None:
        # print(ret, end='\n')
        return False
    return True


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        scores,
        track_ids,
        use_normalized_coordinates=False,
        line_thickness=4):
    scores = np.array(scores)
    box_to_display_str_map = collections.defaultdict(list)

    for i in range(len(boxes)):
        box = tuple(boxes[i])
        # box_to_track_ids_map[box].append(track_ids[i])
        display_str = str('ID {}'.format(track_ids[i]))
        box_to_display_str_map[box].append(display_str)

    for item in boxes:
        box = tuple(item)
        ymin, xmin, ymax, xmax = box
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color='red',
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)

    return None


# def visualize_boxes_and_labels_on_image_array(
#     image,
#     boxes,
#     classes,
#     scores,
#     category_index,
#     instance_masks=None,
#     instance_boundaries=None,
#     keypoints=None,
#     track_ids=None,
#     use_normalized_coordinates=False,
#     max_boxes_to_draw=20,
#     min_score_thresh=.5,
#     agnostic_mode=False,
#     line_thickness=4,
#     groundtruth_box_visualization_color='black',
#     skip_scores=False,
#     skip_labels=False,
#     skip_track_ids=False):
#   """Overlay labeled boxes on an image with formatted scores and label names.
#
#   This function groups boxes that correspond to the same location
#   and creates a display string for each detection and overlays these
#   on the image. Note that this function modifies the image in place, and returns
#   that same image.
#
#   Args:
#     image: uint8 numpy array with shape (img_height, img_width, 3)
#     boxes: a numpy array of shape [N, 4]
#     classes: a numpy array of shape [N]. Note that class indices are 1-based,
#       and match the keys in the label map.
#     scores: a numpy array of shape [N] or None.  If scores=None, then
#       this function assumes that the boxes to be plotted are groundtruth
#       boxes and plot all boxes as black with no classes or scores.
#     category_index: a dict containing category dictionaries (each holding
#       category index `id` and category name `name`) keyed by category indices.
#     instance_masks: a numpy array of shape [N, image_height, image_width] with
#       values ranging between 0 and 1, can be None.
#     instance_boundaries: a numpy array of shape [N, image_height, image_width]
#       with values ranging between 0 and 1, can be None.
#     keypoints: a numpy array of shape [N, num_keypoints, 2], can
#       be None
#     track_ids: a numpy array of shape [N] with unique track ids. If provided,
#       color-coding of boxes will be determined by these ids, and not the class
#       indices.
#     use_normalized_coordinates: whether boxes is to be interpreted as
#       normalized coordinates or not.
#     max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
#       all boxes.
#     min_score_thresh: minimum score threshold for a box to be visualized
#     agnostic_mode: boolean (default: False) controlling whether to evaluate in
#       class-agnostic mode or not.  This mode will display scores but ignore
#       classes.
#     line_thickness: integer (default: 4) controlling line width of the boxes.
#     groundtruth_box_visualization_color: box color for visualizing groundtruth
#       boxes
#     skip_scores: whether to skip score when drawing a single detection
#     skip_labels: whether to skip label when drawing a single detection
#     skip_track_ids: whether to skip track id when drawing a single detection
#
#   Returns:
#     uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
#   """
#   # Create a display string (and color) for every box location, group any boxes
#   # that correspond to the same location.
#   box_to_display_str_map = collections.defaultdict(list)
#   box_to_color_map = collections.defaultdict(str)
#   box_to_instance_masks_map = {}
#   box_to_instance_boundaries_map = {}
#   box_to_keypoints_map = collections.defaultdict(list)
#   box_to_track_ids_map = {}
#   if not max_boxes_to_draw:
#     max_boxes_to_draw = boxes.shape[0]
#   for i in range(min(max_boxes_to_draw, boxes.shape[0])):
#     if scores is None or scores[i] > min_score_thresh:
#       box = tuple(boxes[i].tolist())
#       if instance_masks is not None:
#         box_to_instance_masks_map[box] = instance_masks[i]
#       if instance_boundaries is not None:
#         box_to_instance_boundaries_map[box] = instance_boundaries[i]
#       if keypoints is not None:
#         box_to_keypoints_map[box].extend(keypoints[i])
#       if track_ids is not None:
#         box_to_track_ids_map[box] = track_ids[i]
#       if scores is None:
#         box_to_color_map[box] = groundtruth_box_visualization_color
#       else:
#         display_str = ''
#         if not skip_labels:
#           if not agnostic_mode:
#             if classes[i] in six.viewkeys(category_index):
#               class_name = category_index[classes[i]]['name']
#             else:
#               class_name = 'N/A'
#             display_str = str(class_name)
#         if not skip_scores:
#           if not display_str:
#             display_str = '{}%'.format(int(100*scores[i]))
#           else:
#             display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
#         if not skip_track_ids and track_ids is not None:
#           if not display_str:
#             display_str = 'ID {}'.format(track_ids[i])
#           else:
#             display_str = '{}: ID {}'.format(display_str, track_ids[i])
#         box_to_display_str_map[box].append(display_str)
#         if agnostic_mode:
#           box_to_color_map[box] = 'DarkOrange'
#         elif track_ids is not None:
#           prime_multipler = _get_multiplier_for_color_randomness()
#           box_to_color_map[box] = STANDARD_COLORS[
#               (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
#         else:
#           box_to_color_map[box] = STANDARD_COLORS[
#               classes[i] % len(STANDARD_COLORS)]
#
#   # Draw all boxes onto image.
#   for box, color in box_to_color_map.items():
#     ymin, xmin, ymax, xmax = box
#     if instance_masks is not None:
#       draw_mask_on_image_array(
#           image,
#           box_to_instance_masks_map[box],
#           color=color
#       )
#     if instance_boundaries is not None:
#       draw_mask_on_image_array(
#           image,
#           box_to_instance_boundaries_map[box],
#           color='red',
#           alpha=1.0
#       )
#     draw_bounding_box_on_image_array(
#         image,
#         ymin,
#         xmin,
#         ymax,
#         xmax,
#         color=color,
#         thickness=line_thickness,
#         display_str_list=box_to_display_str_map[box],
#         use_normalized_coordinates=use_normalized_coordinates)
#     if keypoints is not None:
#       draw_keypoints_on_image_array(
#           image,
#           box_to_keypoints_map[box],
#           color=color,
#           radius=line_thickness / 2,
#           use_normalized_coordinates=use_normalized_coordinates)
#
#   return image


def count_people_active_in_video(
        image,
        selected_boxes,
        num_comein,
        num_comeout):
    return (num_comein, num_comeout)
