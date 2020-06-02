# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import tensorflow as tf
import InputManager
import Object_detection_image
import Object_detection_video
import Object_detection_webcam

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', help='Is image, video or webcam',
                    default='webcam')
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='Sample_TFLite_model')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default=0)
parser.add_argument('--image', help='Name of the image file',
                    default=0)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
source = args.source
VIDEO_NAME = args.video
IMAGE_NAME = args.image
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# 1. Input
# testing data
VIDEO_NAME = "video/test1.mp4"
IMAGE_NAME = "pictures/test1.jpg"

if (source == "video"):
    Object_detection_video.detect(VIDEO_NAME)

elif (source == "image"):
    Object_detection_image.detect(IMAGE_NAME)

elif (source == "webcam"):
    Object_detection_webcam.detect()
