
# coding: utf-8

import os
import json
import time

import numpy as np
from skimage import io
from skimage import measure


# Convert label image to keypoint positions.
def mask_to_keypoints(mask):

    keypoints = np.zeros((0, ))

    for idx in range(1, 15):

        # Create mask for each keypoints.
        mask_keypoint = np.int32(mask == idx)
        mask_label = measure.label(mask_keypoint, connectivity=2)
        props = measure.regionprops(mask_label)

        # keypoint not detected.
        if len(props) == 0:
            keypoint = np.zeros((2,))
            keypoints = np.concatenate(
                (keypoints, keypoint[::-1], np.zeros((1, ))), axis=0)
        # Only one region, use the centroid of the region
        elif len(props) == 1:
            keypoint = props[0]['centroid']
            keypoints = np.concatenate(
                (keypoints, keypoint[::-1], np.ones((1, ))), axis=0)
        # Multiple regions, find the region with the largest area
        else:
            areas = [prop['area'] for prop in props]
            idx = np.argmax(areas)
            keypoint = props[idx]['centroid']
            keypoints = np.concatenate(
                (keypoints, keypoint[::-1], np.ones((1, ))), axis=0)

    return keypoints


# Main function: make json file from label images.
def make_json(SSD_json, prediction_dir, output_json):

    # Load SSD detection JSON file.
    SSD_dict = json.load(open(SSD_json, 'r'))
    files = SSD_dict.keys()
    predictions = []

    # Iterative over all the label images.
    start = time.time()
    for i, file in enumerate(files):

        # Print Status.
        if (i + 1) % 1000 == 0 or (i + 1) == len(files):
            print(str(i + 1) + ' / ' + str(len(files)))

        # Load bboxes.
        bboxes = np.array(SSD_dict[file]['bboxes'])
        move = np.array(SSD_dict[file]['move'])

        prediction = dict()
        prediction['image_id'] = file.split('.')[0]
        prediction['keypoint_annotations'] = dict()

        # Iterate over each human in the image.
        for idx in range(bboxes.shape[0]):

            # Read label image.
            prediction_mask = prediction_dir + '/' + \
                prediction['image_id'] + '_' + str(idx) + '.png'
            if os.path.isfile(prediction_mask):
                mask = io.imread(prediction_mask)
            else:
                continue

            # Convert label image to keypoint positions.
            keypoints = mask_to_keypoints(mask)
            keypoints[0::3] += move[idx, 0]
            keypoints[1::3] += move[idx, 1]

            # Store in prediction format when there is more than 4 visible keypoints.
            if sum(keypoints[2::3] != 0) >= 4:
                key = 'human' + str(idx + 1)
                prediction['keypoint_annotations'][key] = np.int32(
                    keypoints).tolist()
            else:
                continue

        # Append for each image.
        predictions.append(prediction)

    print('Successfully processed all the images in %.2f seconds.' %
          (time.time() - start))

    # Dump prediction JSON file.
    start = time.time()
    json.dump(predictions, open(output_json, 'w'))
    print('Successfully generated JSON prediction file in %.2f seconds.' %
          (time.time() - start))

    return predictions
