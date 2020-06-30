import numpy as np
import cv2
import argparse

""" Functions for drawing hand template_sets. """


def make_drawing(hand_keypoints, keypoint_confidence_threshold, size=500):
    mask = np.zeros((size, size))
    hand_base_x, hand_base_y = hand_keypoints[0, :2]
    x_old, y_old = hand_base_x, hand_base_y

    i = 1
    while i < len(hand_keypoints):
        if not (hand_keypoints[i, 2] <= keypoint_confidence_threshold):
            x, y = hand_keypoints[i, :2]
            color = (255, 255, 255)
            cv2.line(mask, (int(x_old), int(y_old)), (int(x), int(y)), color,
                     thickness=4)
        i += 1
        if (i-1) % 4 == 0:
            x_old, y_old = hand_base_x, hand_base_y
        else:
            x_old, y_old = x, y
    return mask


def draw_hand(hand_keypoints, keypoint_confidence_treshold=0.15, info_text="Hand keypoints"):
    res = make_drawing(hand_keypoints, keypoint_confidence_treshold)
    cv2.imshow(info_text, res)
    cv2.waitKey(0)


def save_keypoints_as_img(hand_keypoints, filename, keypoint_confidence_treshold=0.15):
    res = make_drawing(hand_keypoints, keypoint_confidence_treshold)
    cv2.imwrite(str(filename), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_keypoints_path")
    args = parser.parse_known_args()

    keypoints = np.load(args[0].json_keypoints_path, allow_pickle=True)
    draw_hand(keypoints)
    # save_keypoints_as_img(keypoints, "A_template.jpg")
