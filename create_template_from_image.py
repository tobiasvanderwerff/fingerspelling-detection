#!/usr/bin/env python3

import sys
import cv2
import os
import argparse
import numpy as np
import draw_hand
import util
from pipeline import SegmentDetectionPipeline
from op1 import OpenPose
from pathlib import Path


'''
Create and save an array of coordinates from a template alphabet letter image
which shows the sign in the right hand. Requires OpenPose Python API to run.
'''

try:
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("-o", "--output_path",
                        help="Path where to store the template.")
    args = parser.parse_known_args()

    imageToProcess = cv2.imread(args[0].image_path)
    params = util.prepare_openpose_params(args)
    op_wrapper = OpenPose(params)
    op_wrapper.start()
    datum = op_wrapper.process(imageToProcess)

    cv2.imshow("Keypoints for template", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)

right_hand_keypoints = np.array(datum.handKeypoints[1][0])  # shape for one person detected: (2, 1, 21, 3)
right_hand_keypoints_norm = SegmentDetectionPipeline.normalize(right_hand_keypoints)

create_template = input("Create template? [Y/n]\n")

if not (create_template == "N" or create_template == "n" or
        create_template == "No" or create_template == "no"):
    filename = str(Path(args[0].output_path) / (os.path.basename(os.path.splitext(args[0].image_path)[0] + "_template")))
    np.save(filename, right_hand_keypoints_norm)
    draw_hand.draw_hand(right_hand_keypoints_norm, info_text="Template result")
    print("\nTemplate created in " + filename + ".npy")
