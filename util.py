import os
import numpy as np
from pathlib import Path


def add_path_args(args, params=None):
    if params is None:
        params = dict()
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item
    return params


def prepare_openpose_params(args):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../openpose/models/"
    params["model_pose"] = "BODY_25"     
    params["num_gpu"] = 1
    params["keypoint_scale"] = 3        # Scale keypoint coordinates to [0,1]
    params["hand"] = True
    params["hand_detector"] = 3         # Enable hand tracking for video and webcam
    params["hand_scale_number"] = 6     # Configuration for better results
    params["hand_scale_range"] = 0.4    # Configuration for better results
    params = add_path_args(args, params)
    return params


def load_templates(template_dir):
    templates = dict()
    for filename in os.listdir(template_dir):
        template = np.load(Path(template_dir) / filename, allow_pickle=True)
        templates.update({filename: template})
    return templates
