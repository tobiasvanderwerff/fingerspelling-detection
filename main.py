#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from src.segment_detection import SegmentDetector, Video
from src.results_view import summarize_results, show_predicted_segments
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to video or video directory to detect fingerspelling segments for.")
parser.add_argument("--op_out", help="Directory containing OpenPose output in json format. Each video should have its own directory within this directory.")
parser.add_argument("--template_dir", default="template_sets/generic", help="Directory containing the template_sets.")
parser.add_argument("--eaf_annot_dir", default="../NGT_corpus/CNGT_GlossTiers_Fingerspelling", help="Path where EAF annotation files are stored, for evaluation on Corpus NGT.")
parser.add_argument("--eps", default=7, type=int)
parser.add_argument("--min_samples", default=5, type=int)
parser.add_argument("--movement_threshold", default=0.025, type=float)
parser.add_argument("--template_match_thresh", default=56000, type=int)
args = parser.parse_args()

input_path = Path(args.input)
eaf_annot_dir = Path(args.eaf_annot_dir)
op_output_dir = Path(args.op_out)
template_dir = Path(args.template_dir)
eps = args.eps
min_samples = args.min_samples
movement_threshold = args.movement_threshold
template_match_thresh = args.template_match_thresh

recall_per_video1 = {"s1": [], "s2": []}
recall_per_video2 = {"s1": [], "s2": []}
precision_per_video1 = {"s1": [], "s2": []}
precision_per_video2 = {"s1": [], "s2": []}
template_match_cnt = Counter()
no_match_cnt, preselection_rejection_cnt, total_n_frames = 0, 0, 0
if input_path.is_dir():  # Evaluation for Corpus NGT. Assumes two subfolders: s1 and s2.
    for s in ["s1", "s2"]:
        for video_path in os.listdir(input_path / s):
            input_video = Video(input_path / s / video_path, op_output_dir, corpus_ngt=True)
            detector = SegmentDetector(input_video, op_output_dir, template_dir, eps=eps,
                                       min_samples=min_samples, movement_threshold=movement_threshold,
                                       template_match_thresh=template_match_thresh)
            detector.detect_segments()
            detector.evaluate_results(eaf_annot_dir, show_results=True)

            precision_per_video1[s].append(detector.results["precision1"])
            recall_per_video1[s].append(detector.results["recall1"])
            precision_per_video2[s].append(detector.results["precision2"])
            recall_per_video2[s].append(detector.results["recall2"])
            no_match_cnt += detector.results["no_match_cnt"]
            preselection_rejection_cnt += detector.results["preselection_rejection_cnt"]
            template_match_cnt += detector.results["template_match_cnt"]
            total_n_frames += input_video.n_frames
    summarize_results(recall_per_video1, precision_per_video1, recall_per_video2, precision_per_video2,
                      template_match_cnt, no_match_cnt, preselection_rejection_cnt, total_n_frames, eps, min_samples,
                      movement_threshold, template_match_thresh, input_path.name, template_dir.name)
else:  # Single video
    input_video = Video(input_path, op_output_dir, corpus_ngt=False)
    detector = SegmentDetector(input_video, op_output_dir, template_dir, eps=eps, min_samples=min_samples,
                               movement_threshold=movement_threshold, template_match_thresh=template_match_thresh)
    detector.detect_segments()
    show_predicted_segments(detector.results["predicted_segments"], detector.video.fps)
