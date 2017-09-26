
# coding: utf-8

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw


# Keypoint pairs for drawing lines.
line_list = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8],
                      [9, 10], [10, 11], [12, 13], [0, 13], [3, 13], [0, 6], [3, 9], [6, 9]])


# Main function.
def show(image_dir, prediction_json, idx=None, save_path=None):

    # Load predictions and images.
    predictions = json.load(open(prediction_json, 'r'))

    # If index is not specified, then pick the index random.
    if idx is None:
        idx = random.randint(0, len(predictions) - 1)

    try:
        # Read image and keypoints.
        img = Image.open(image_dir + predictions[idx]['image_id'] + '.jpg')
        keypoint_predictions = predictions[idx]['keypoint_annotations']
        print(image_dir + predictions[idx]['image_id'] + '.jpg')
    except:
        print('File %s doesn\'t exist.' % predictions[idx]['image_id'])

    draw = ImageDraw.Draw(img)

    # For every human.
    for key in keypoint_predictions.keys():

        keypoints = np.reshape(keypoint_predictions[key], (14, 3))

        # Draw keypoints.
        for i in range(14):
            if keypoints[i, 2] == 1:
                draw.rectangle([keypoints[i, 0] - 5, keypoints[i, 1] - 5,
                                keypoints[i, 0] + 5, keypoints[i, 1] + 5],
                               outline=(0, 0, 0), fill=(0, 0, 0))

        # Draw lines between keypoints.
        for i in range(line_list.shape[0]):
            if keypoints[line_list[i, 0], 2] == 1 and keypoints[line_list[i, 1], 2] == 1:
                draw.line([(keypoints[line_list[i, 0], 0], keypoints[line_list[i, 0], 1]),
                           (keypoints[line_list[i, 1], 0], keypoints[line_list[i, 1], 1])],
                           width=4, fill=(255, 0, 0))

    # Show the drawing image.
    img.show()

    # If save path is specified, then save the drawing image.
    if save_path is not None:
        img.save(save_path)
