#!/usr/bin/env python3

import cv2
import argparse
from src import util
import json
import os
from src.op1 import OpenPose
from pathlib import Path

""" Process a video with OpenPose and store the results in json format for processing
    by the fingerspelling detection pipeline."""


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--video_path", help="Process a video.")
parser.add_argument("--write_json", help="Dir to store json output in.")
args = parser.parse_known_args()

params = util.prepare_openpose_params(args)
op = OpenPose(params)
op.start()

out_dir = args[0].write_json + "/" + os.path.basename(args[0].video_path)
Path(out_dir).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(args[0].video_path)
ret, frame = cap.read()
frame_no = 1
while ret:
    datum = op.process(frame)

    json_out = {"frame_no": frame_no, "people": [{}]}
    pose_keypoints = datum.poseKeypoints.flatten().tolist()
    left_hand_keypoints = datum.handKeypoints[0].flatten().tolist()
    right_hand_keypoints = datum.handKeypoints[1].flatten().tolist()
    if len(pose_keypoints) > 1:
        json_out["people"][0].update({"pose_keypoints_2d": pose_keypoints})
    else:
        json_out["people"][0].update({"pose_keypoints_2d": []})
    if len(left_hand_keypoints) > 1:
        json_out["people"][0].update({"hand_left_keypoints_2d": left_hand_keypoints})
    else:
        json_out["people"][0].update({"hand_left_keypoints_2d": []})
    if len(right_hand_keypoints) > 1:
        json_out["people"][0].update({"hand_right_keypoints_2d": right_hand_keypoints})
    else:
        json_out["people"][0].update({"hand_right_keypoints_2d": []})
    with open("{}/{}_{}.json".format(out_dir, os.path.basename(args[0].video_path), str(frame_no).zfill(10)), "w") as fd:
        fd.write(json.dumps(json_out))

    ret, frame = cap.read()
    frame_no += 1
cap.release()
cv2.destroyAllWindows()
