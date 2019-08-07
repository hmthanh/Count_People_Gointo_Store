# Import packages
import collections
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from myutils.box_pruning import pruning_overlap_box_on_score
from myutils.const import Const
from myutils.draw_box_on_image import visualize_boxes_and_labels_on_image_array as vis_box, count_people_active_in_video
from tracker.tracker import Tracker
# Import utilites
from utils import label_map_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'abc.jpg'
INPUT_NAME = 'goin.mp4'
# OUTPUT_NAME = 'output.mp4'
OUTPUT_NAME = 'output.avi'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
PATH_TO_INPUT = os.path.join(CWD_PATH, INPUT_NAME)
PATH_TO_OUTPUT = os.path.join(CWD_PATH, OUTPUT_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Tracker define
TRACKER_LIST_COOR = []
TRACKER_BOX_MAP = collections.defaultdict(list)

max_age = 15
min_hits = 1

isVideo = True
isRender = True

'''
START VARIABLE INIT
Config variable count number of people come in or come out
'''
# Load image using OpenCV and
if isVideo:
    video = cv2.VideoCapture(PATH_TO_INPUT)
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    fm_height, fm_width, fm_chanel = frame.shape
else:
    frame = cv2.imread(PATH_TO_IMAGE)
    frame_expanded = np.expand_dims(frame, axis=0)
    fm_height, fm_width, fm_chanel = frame.shape

# Counting variable
frame_counter = 0
num_come_out = 0
num_come_in = 0
status = ""

# Text init
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.3
color = (0, 0, 255)
stroke = 2
text_h = 45
coord_w = 30
coord_h = fm_height

# Video output encoding settings
if isVideo and isRender:
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 25.0
    video_writer = cv2.VideoWriter('newoutput.mp4', fourcc, fps, (fm_width, fm_height))
'''
END INIT
'''

''' START TENSORFLOW SETUP '''
# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
''' END TENSORFLOW SETUP '''

Const.set_resolution(fm_width, fm_height)
START_DOOR = Const.get_start()
END_DOOR = Const.get_end()

track_ids = []

if isVideo:
    while video.isOpened():
        frame_counter += 1
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        boxes, scores = pruning_overlap_box_on_score(
            boxes,
            scores,
            min_score_thresh=0.8)
        TRACKER_BOX_MAP, track_boxes, track_ids, num_come_in, num_come_out = Tracker.assign_to_tracker(
            boxes,
            TRACKER_BOX_MAP,
            track_ids,
            num_come_in,
            num_come_out)

        # Draw the results of the detection
        selected_boxes = vis_box(
            frame,
            track_boxes,
            scores,
            track_ids,
            use_normalized_coordinates=True,
            line_thickness=8)

        # Draw text to video frame
        str_frame = "Frame : " + str(frame_counter)
        str_come_in = "Come in : " + str(num_come_in)
        str_come_out = "Come out : " + str(num_come_out)
        display_list_str = (str_frame, str_come_in, str_come_out)
        coord_h = fm_height
        for display_str in display_list_str:
            coord_h -= text_h
            cv2.putText(frame, display_str, (coord_w, coord_h), font, font_size, color, stroke, cv2.LINE_AA)

        y_door_min, x_door_min = START_DOOR
        y_door_max, x_door_max = END_DOOR
        cv2.rectangle(frame, (y_door_min, x_door_min), (y_door_max, x_door_max), (255, 255, 0), 5)

        # Write each frame to the output
        if isRender:
            video_writer.write(frame)

        # Resize image with resolution
        im_width_resized = int(fm_width * 0.5)
        im_height_resized = int(fm_height * 0.5)
        frame_resized = cv2.resize(frame, (im_width_resized, im_height_resized), interpolation=cv2.INTER_AREA)

        # All the results have been drawn on image. Now display the image.
        cv2.imshow('Object detector', frame_resized)

        # Press any key to close the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    boxes, scores = pruning_overlap_box_on_score(boxes, scores, min_score_thresh=0.8)
    # Draw the results of the detection
    selected_boxes = vis_box(
        frame,
        boxes,
        scores,
        track_ids=[1, 2, 3, 4, 5, 6, 7],
        use_normalized_coordinates=True,
        line_thickness=8)

    # Draw text to video frame
    str_frame = "Frame : " + str(frame_counter)
    str_come_in = "Come in : " + str(num_come_in)
    str_come_out = "Come out : " + str(num_come_out)
    display_list_str = (str_frame, str_come_in, str_come_out)
    coord_h = fm_height
    for display_str in display_list_str:
        coord_h -= text_h
        cv2.putText(frame, display_str, (coord_w, coord_h), font, font_size, color, stroke, cv2.LINE_AA)
    cv2.rectangle(frame, START_DOOR, END_DOOR, (255, 255, 0), 5)

    # Resize image with resolution
    im_width_resized = int(fm_width * 0.5)
    im_height_resized = int(fm_height * 0.5)
    frame_resized = cv2.resize(frame, (im_width_resized, im_height_resized), interpolation=cv2.INTER_AREA)

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', frame_resized)

    # Press any key to close the video
    cv2.waitKey(0)

# Clean up
if isVideo:
    video.release()

if isRender:
    video_writer.release()

cv2.destroyAllWindows()
