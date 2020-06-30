#!/usr/bin/env python3

import cv2
import os
import numpy as np
from src import util
import re
from pathlib import Path
from src.eaf_parsing import EAFParser, ms_to_frame_no
from src.pipeline import SegmentDetectionPipeline
from collections import Counter
from typing import List, Tuple
from src.results_view import show_video_results


def precision(tp, fp):
    if tp + fp != 0:
        return tp / (tp + fp)
    else:
        return 0


def recall(tp, fn):
    if tp + fn != 0:
        return tp / (tp + fn)
    else:
        return 0


def metric1(n_correct_segments, n_true_segments, n_clusters):
    precision1 = min(1, n_correct_segments / max(1, (n_clusters - 1)))
    recall1 = n_correct_segments / max(1, n_true_segments)
    return precision1, recall1


def metric2(ground_truth, predicted_segments):
    tp = int(np.sum(ground_truth * predicted_segments))
    fp = int(np.sum(np.logical_not(ground_truth) * predicted_segments))
    fn = int(np.sum(ground_truth * np.logical_not(predicted_segments)))
    return precision(tp, fp), recall(tp, fn), tp, fp, fn


class Video:
    def __init__(self, path: Path, op_output_dir_path, corpus_ngt):
        self.path = path
        self.fps = self.video_fps(corpus_ngt)
        self.n_frames = self.no_of_op_frames(op_output_dir_path)
        self.signer = re.search(r"S\d{3}", path.name).group()

    def video_fps(self, corpus_ngt=False) -> int:
        if corpus_ngt:
            return 25
        else:
            cap = cv2.VideoCapture(str(self.path))
            res = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return res

    def no_of_op_frames(self, op_output_dir_path):
        """ For counting the number of frames of the video, we count the number of files that OpenPose produced for the
            video (OpenPose produces one JSON file per frame)."""
        return len(os.listdir(op_output_dir_path / self.path.stem))


class SegmentDetector:
    def __init__(self, video: Video, op_output_dir, template_dir, **parameters):
        self.op_output_dir = op_output_dir
        self.templates = util.load_templates(template_dir)
        self.parameters = parameters
        self.video = video
        self.pipeline = SegmentDetectionPipeline(parameters["movement_threshold"])
        self.results = {}

    def detect_segments(self):
        matches, no_match_cnt, preselection_rejection_cnt, template_match_cnt, predicted_segments, n_clusters = \
            self.pass_video_through_pipeline(include_preselection=True)
        self.results.update({"predicted_segments": predicted_segments, "n_clusters": n_clusters,
                             "no_match_cnt": no_match_cnt, "preselection_rejection_cnt": preselection_rejection_cnt,
                             "template_match_cnt": template_match_cnt})

    def evaluate_results(self, eaf_annot_dir, show_results=True):
        predicted_segments = self.results["predicted_segments"]
        n_clusters = self.results["n_clusters"]
        ground_truth, fingerspell_segments, n_true_segments = self.prepare_ground_truth(eaf_annot_dir)
        n_correct_segments = self.count_correctly_detected_segments(fingerspell_segments, predicted_segments)
        pr1, rec1 = metric1(n_correct_segments, n_true_segments, n_clusters)
        pr2, rec2, _, _, _ = metric2(ground_truth, predicted_segments)
        self.results.update({"precision1": pr1, "recall1": rec1})
        self.results.update({"precision2": pr2, "recall2": rec2})
        pr1, rec1 = metric1(n_correct_segments, n_true_segments, n_clusters)
        pr2, rec2, tp, fp, fn = metric2(ground_truth, predicted_segments)
        if show_results:
            show_video_results(self.video.path, self.video.signer, n_correct_segments, len(fingerspell_segments),
                               n_clusters, pr1, rec1, pr2, rec2, tp, fp, fn)

    def prepare_ground_truth(self, eaf_annot_dir):
        ground_truth = [0] * self.video.n_frames
        eaf_fn = re.search(r"CNGT\d{4}", self.video.path.name).group() + ".eaf"
        eaf_parser = EAFParser(eaf_annot_dir / eaf_fn, self.video.signer)
        fingerspell_timeslots = eaf_parser.parse_fingerspelling_timeslots(self.parameters["min_samples"],
                                                                          self.video.fps)
        fingerspell_segments = self.merge_close_fingerspelling_timeslots(
            [(ms_to_frame_no(s, self.video.fps), ms_to_frame_no(e, self.video.fps)) for s, e in fingerspell_timeslots])
        for begin_frame, end_frame in fingerspell_segments:
            for i in range(begin_frame, end_frame + 1):
                ground_truth[i] = 1
        return ground_truth, fingerspell_segments, len(fingerspell_segments)

    def pass_video_through_pipeline(self, include_preselection=True):
        frame_no, no_match_cnt, preselection_rejection_cnt = 1, 0, 0
        template_match_cnt = Counter()
        matches = np.full(self.video.n_frames, -10, dtype=int)
        for frame in sorted(os.listdir(self.op_output_dir / self.video.path.stem)):
            self.pipeline.load_json_keypoints(self.op_output_dir / self.video.path.stem / frame)
            preselection_passed = self.pipeline.preselection(include_preselection)
            if preselection_passed:
                right_hand_keypoints_norm = SegmentDetectionPipeline.normalize(self.pipeline.prev_right_hand_keypoints)
                best_score, match_name = self.pipeline.best_match(right_hand_keypoints_norm, self.templates)
                if not (best_score > self.parameters["template_match_thresh"] or best_score == -1):
                    template_match_cnt[match_name] += 1
                    matches[frame_no] = frame_no
                else:
                    no_match_cnt += 1
            else:
                preselection_rejection_cnt += 1
            frame_no += 1
        predicted_segments, n_clusters = self.pipeline.dbscan(matches, self.parameters["min_samples"],
                                                              self.parameters["eps"], self.video.n_frames)
        return matches, no_match_cnt, preselection_rejection_cnt, template_match_cnt, predicted_segments, n_clusters

    def count_correctly_detected_segments(self, fingerspell_segments: List[Tuple], predicted_segments: List[int]):
        n_correct_segments = 0
        for begin_frame, end_frame in fingerspell_segments:
            n_segment_frames = end_frame - begin_frame
            req_perc_detected = 0.20
            i = max(1, begin_frame - 2 * self.video.fps)
            n_detected_frames, predicted_segment_started = 0, False
            while i < len(predicted_segments) and i < end_frame:
                if predicted_segments[i-1] == 0 and predicted_segments[i] == 1:  # Start of segment
                    while predicted_segments[i] == 1 and i < len(predicted_segments) and i < end_frame + 2 * self.video.fps:
                        if begin_frame <= i <= end_frame:
                            n_detected_frames += 1
                        i += 1
                i += 1
            if predicted_segments[i] == 0 and n_detected_frames / n_segment_frames >= req_perc_detected:
                n_correct_segments += 1
        return n_correct_segments

    def save_matching_frames(self, predicted_segments, ground_truth, out_path, save_nonmatching=False):
        if save_nonmatching:
            def condition(y_hat, y, frame): return y_hat[frame] != 1 and y[frame] == 1
        else:
            def condition(y_hat, y, frame): return y_hat[frame] == 1 and y[frame] == 1
        cap = cv2.VideoCapture(str(self.video.path))
        frame_no = 1
        ret, image = cap.read()
        while ret and frame_no < len(predicted_segments):
            if condition(predicted_segments, ground_truth, frame_no):
                Path(Path(out_path) / self.video.path.stem).mkdir(exist_ok=True)
                cv2.imwrite(str(Path(out_path) / self.video.path.stem / "".join([str(frame_no), ".png"])), image)
            frame_no += 1
            ret, image = cap.read()

    # Fingerspelling sequences that are less than threshold frames apart are merged into one.
    @staticmethod
    def merge_close_fingerspelling_timeslots(timeslots: List[Tuple[int, int]], threshold=3):
        i = 0
        res = []
        timeslots = sorted(timeslots, key=lambda ts: ts[0])
        while i < len(timeslots) - 1:
            res.append(timeslots[i])
            if 0 <= timeslots[i + 1][0] - timeslots[i][1] <= threshold:
                res[-1] = (timeslots[i][0], timeslots[i + 1][1])
                i += 1
            i += 1
        if i == len(timeslots) - 1:
            res.append(timeslots[i])
        return res


