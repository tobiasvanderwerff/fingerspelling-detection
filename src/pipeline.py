import json
import numpy as np
import functools
import time
from math import sqrt
from typing import Tuple, List
from sklearn.cluster import dbscan

""" All the classes and functions for the segment detection pipeline, as laid out in the thesis. """

# Keypoints are of the form (x, y, confidence).
Keypoint = Tuple[float, float, float]


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.counter += 1
        start_time = time.time()
        res = func(*args, **kwargs)
        wrapper.total_time += time.time() - start_time
        print("Function {} was called {}x taking a total of {} seconds".format(func.__name__, wrapper.counter,
                                                                               wrapper.total_time))
        return res
    wrapper.total_time = 0
    wrapper.counter = 0
    return wrapper


def euclidean_distance(x1, x2, y1, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def constraint1(right_hand_keypoints: List[Keypoint]) -> bool:
    """ Were hand keypoints detected by OpenPose? Input array is a numpy array. """
    return len(right_hand_keypoints) > 0 and any((right_hand_keypoints != 0).flatten())


def constraint2(lowest_right_hand_keypoint: int, right_elbow_y_coord: int) -> bool:
    """ Is lowest right hand keypoint higher than right elbow? (note that top left coordinate of frame has
        coordinates (0, 0)). """
    return lowest_right_hand_keypoint < right_elbow_y_coord


def constraint3(lowest_right_hand_keypoint: int, lowest_left_hand_keypoint: int) -> bool:
    """ Is lowest right hand keypoint higher than lowest left hand keypoint? """
    return lowest_right_hand_keypoint < lowest_left_hand_keypoint


def constraint4(right_hand_kps: List[Keypoint], prev_right_hand_kps: List[Keypoint], movement_threshold) -> bool:
    """ Is the distance between the current hand keypoints and the hand keypoints in the previous frame smaller
        than movement_threshold? """
    if prev_right_hand_kps.size == 0:
        return True
    return euclidean_distance(right_hand_kps[0][0], prev_right_hand_kps[0][0], right_hand_kps[0][1],
                              prev_right_hand_kps[0][1]) < movement_threshold


class SegmentDetectionPipeline:
    json_keypoints = None
    prev_right_hand_keypoints = np.array([])

    def __init__(self, movement_threshold):
        self.movement_threshold = movement_threshold

    def load_json_keypoints(self, json_path):
        with open(json_path, "r") as fd:
            self.json_keypoints = json.load(fd)
        if len(self.json_keypoints["people"]) == 0:
            self.json_keypoints["people"].append({"pose_keypoints_2d": [], "hand_right_keypoints_2d": [],
                                                  "hand_left_keypoints_2d": []})

    def lowest_left_hand_keypoint(self):
        """ Returns y coordinate of lowest left hand keypoint, given self.json_keypoints.
            If left hand keypoints were not detected by OpenPose, return the lowest y coordinate.
            Call load_json_keypoints before calling this function. """
        left_hand_keypoints = np.reshape(self.json_keypoints["people"][0]["hand_left_keypoints_2d"], (-1, 3))
        if len(left_hand_keypoints) > 0 and any(np.array(self.json_keypoints["people"][0]["hand_left_keypoints_2d"]) != 0):
            lowest_left_hand_keypoint = max(left_hand_keypoints[:, 1])
        else:
            lowest_left_hand_keypoint = 1
        return lowest_left_hand_keypoint

    # @timer
    def preselection(self, include_physical_constraints=True) -> bool:
        """ Checks if keypoints meet the preselection criteria. Returns true if all constraints are met.
            Call load_json_keypoints before calling this function. """
        preselection_passed = False
        # Assumption is that one person is visible in the frame.
        right_hand_keypoints = np.reshape(self.json_keypoints["people"][0]["hand_right_keypoints_2d"], (-1, 3))
        cnstr1 = constraint1(right_hand_keypoints)
        if cnstr1:
            if include_physical_constraints:
                right_elbow_y_coord = np.reshape(self.json_keypoints["people"][0]["pose_keypoints_2d"], (-1, 3))[3, 1]
                lowest_right_hand_keypoint = max(right_hand_keypoints[:, 1])
                cnstr2 = constraint2(lowest_right_hand_keypoint, right_elbow_y_coord)
                cnstr3 = constraint3(lowest_right_hand_keypoint, self.lowest_left_hand_keypoint())
                cnstr4 = constraint4(right_hand_keypoints, self.prev_right_hand_keypoints, self.movement_threshold)
                if cnstr2 and cnstr3 and cnstr4:
                    preselection_passed = True
            else:
                preselection_passed = True
        self.prev_right_hand_keypoints = right_hand_keypoints
        return preselection_passed

    @classmethod
    def best_match(cls, normalized_keypoints: List[Keypoint], templates):
        scores = cls._template_matching(normalized_keypoints, templates)
        best_matches = sorted(scores.items(), key=lambda x: x[1])
        match_name, lowest_score = best_matches[0]
        return int(lowest_score), match_name

    @staticmethod
    # @timer
    def normalize(right_hand_keypoints: List[Keypoint], size=500):
        # Shift keypoints such that minimum keypoint is at (0, 0).
        min_x_coordinate_right_hand = min(right_hand_keypoints[:, 0])
        min_y_coordinate_right_hand = min(right_hand_keypoints[:, 1])
        shifted_keypoints = (right_hand_keypoints -
                             [min_x_coordinate_right_hand, min_y_coordinate_right_hand, 0])
        max_x_coordinate_right_hand = max(shifted_keypoints[:, 0])
        max_y_coordinate_right_hand = max(shifted_keypoints[:, 1])

        # Stretch coordinates to frame with depth = height = size.
        keypoints_norm = (shifted_keypoints *
                          [size / max_x_coordinate_right_hand, size / max_y_coordinate_right_hand, 1])
        return keypoints_norm

    @staticmethod
    # @timer
    def _template_matching(right_hand_keypoints: List[Keypoint], templates, confidence_threshold=0.15):
        losses = dict()
        for filename, template in templates.items():
            sum_of_squared_euclidean = 0
            reliable_keypoints = 0
            # We exclude the base keypoint from the comparison.
            for pa, pb in zip(template[1:], right_hand_keypoints[1:]):
                if not (pa[2] <= confidence_threshold or pb[2] <= confidence_threshold):
                    sum_of_squared_euclidean += (pa[0] - pb[0])**2 + (pa[1] - pb[1])**2
                    reliable_keypoints += 1
            loss = sum_of_squared_euclidean / reliable_keypoints if (reliable_keypoints != 0) else -1
            losses.update({filename: loss})
        return losses

    @staticmethod
    # @timer
    def dbscan(matches, min_samples, eps, n_frames):
        cluster_labels = dbscan(np.array(matches).reshape(-1, 1), eps=eps, min_samples=min_samples)[1]
        # Note that n_clusters contains an additional superfluous cluster (non-match cluster). So no. of detected
        # segments is n_clusters - 1.
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        segments = np.zeros(n_frames)
        for i in range(1, n_clusters):
            cluster_locations = (cluster_labels == i)
            start_frame = np.argmax(cluster_locations)
            end_frame = len(cluster_locations) - np.argmax(cluster_locations[::-1]) - 1
            for j in range(start_frame, end_frame + 1):
                segments[j] = 1
        return segments, n_clusters
