from typing import List, Dict
from collections import Counter


def format_secs_nicely(secs):
    mins = int(secs // 60)
    secs = secs % 60
    return "{}m {:.3f}s".format(mins, secs)


def show_predicted_segments(predicted_segments, video_fps):
    segment_timeslots = timeslots_from_segments(predicted_segments, video_fps)
    print("{} possible fingerspelling segments detected:".format(len(segment_timeslots)))
    for i, segment in enumerate(segment_timeslots):
        start_time = format_secs_nicely(segment[0])
        end_time = format_secs_nicely(segment[1])
        print("Segment {}: {} - {}".format(i+1, start_time, end_time))


def show_video_results(video_path, video_signer, n_correct_segments, n_true_segments, n_clusters, pr1, rec1, pr2, rec2,
                       tp, fp, fn):
    print("{} ({}): Pr: {:.4f}, Rec: {:.4f} -- TP: {}, FP: {}, FN: {} -- {} segments detected, "
          "{}/{} correctly detected ({:.1f}%)".format(video_path, video_signer, pr1, rec1, tp, fp, fn,
                                                      n_clusters - 1, n_correct_segments, n_true_segments,
                                                      rec1 * 100))


def summarize_results(recall_per_video1: Dict[str, List[float]], precision_per_video1: Dict[str, List[float]],
                      recall_per_video2: Dict[str, List[float]], precision_per_video2: Dict[str, List[float]],
                      template_match_cnt: Counter, no_match_cnt: int, preselection_rejection_cnt: int,
                      total_n_frames: int, eps, min_samples, movement_threshold, template_match_thresh, input_path_name,
                      template_dir_name):
    print("******************************************************")
    print("Input set:\t{}".format(input_path_name))
    print("Template set :\t{}\n".format(template_dir_name))

    print(
        "Parameters:\n"
        "eps: %d\n"
        "min_samples: %d\n"
        "movement_threshold: %.3f\n"
        "template_match_thresh: %d \n" % (eps, min_samples, movement_threshold, template_match_thresh)
    )

    print("Metric 1: Detected segments")
    print("s1 -- AP: %.3f, AR: %.3f" % (sum(precision_per_video1["s1"]) / len(precision_per_video1["s1"]),
                                        sum(recall_per_video1["s1"]) / len(recall_per_video1["s1"])))
    print("s2 -- AP: %.3f, AR: %.3f" % (sum(precision_per_video1["s2"]) / len(precision_per_video1["s2"]),
                                        sum(recall_per_video1["s2"]) / len(recall_per_video1["s2"])))
    print("total -- AP: %.3f, AR: %.3f\n" % ((sum(precision_per_video1["s2"]) + sum(precision_per_video1["s1"])) /
                                             (len(precision_per_video1["s2"]) + len(precision_per_video1["s1"])),
                                             (sum(recall_per_video1["s2"]) + sum(recall_per_video1["s1"])) /
                                             (len(recall_per_video1["s2"]) + len(recall_per_video1["s1"]))))

    print("Metric 2: Detected frames")
    print("s1 -- AP: %.3f, AR: %.3f" % (sum(precision_per_video2["s1"]) / len(precision_per_video2["s1"]),
                                        sum(recall_per_video2["s1"]) / len(recall_per_video2["s1"])))
    print("s2 -- AP: %.3f, AR: %.3f" % (sum(precision_per_video2["s2"]) / len(precision_per_video2["s2"]),
                                        sum(recall_per_video2["s2"]) / len(recall_per_video2["s2"])))
    print("AP: %.3f, AR: %.3f\n" % ((sum(precision_per_video2["s2"]) + sum(precision_per_video2["s1"])) /
                                    (len(precision_per_video2["s2"]) + len(precision_per_video2["s1"])),
                                    (sum(recall_per_video2["s2"]) + sum(recall_per_video2["s1"])) /
                                    (len(recall_per_video2["s2"]) + len(recall_per_video2["s1"]))))

    total_matches = sum(template_match_cnt.values())
    print("Matches per template:")
    for k, v in sorted(template_match_cnt.items(), key=lambda x: x[1]):
        print("{}: {} ({:.2f}%)".format(k, v, int(v / total_matches * 100)))
    print("\nNo of rejected frames (preselection): {} ({}%)".format(preselection_rejection_cnt,
                                                                    int(preselection_rejection_cnt /
                                                                        total_n_frames * 100)))
    print("No of rejected frames (template_match_thresh): {} ({}%)".format(no_match_cnt,
                                                                           int(no_match_cnt / total_n_frames * 100)))

    print("******************************************************\n")


def timeslots_from_segments(segments: List[int], fps):
    """ Returns time slots (start and end of segments) for each segment in the given list.
        Segments is a list of binary numbers indicating for each frame whether it is a segment frame (1) or not (0). """
    segment_timeslots, segment_found = [], False
    ln = len(segments)
    for i, frame_no in enumerate(segments):
        if frame_no == 1:  # Segment frame
            if not segment_found:
                segment_start_idx = i
                segment_found = True
            elif i == ln - 1 or segments[i+1] == 0:
                segment_end_idx = i
                start_time = segment_start_idx / fps
                end_time = segment_end_idx / fps
                segment_timeslots.append((start_time, end_time))
                segment_found = False
    return segment_timeslots
