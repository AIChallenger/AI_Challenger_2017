#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation utility for human skeleton system keypoint task.

This python script is used for calculating the final score (mAP) of the test result,

based on your submited file and the reference file containing ground truth.

usage

python keypoint_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH

A test case is provided, submited file is submit.json, reference file is ref.json, test it by:

python keypoint_eval.py --submit ./keypoint_sample_predictions.json \
                        --ref ./keypoint_sample_annotations.json

The final score of the submited result, error message and warning message will be printed.
"""


import json
import time
import argparse
import pprint

import numpy as np


def load_annotations(anno_file, return_dict):
    """Convert annotation JSON file."""

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                       0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                       0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    try:
        annos = json.load(open(anno_file, 'r'))
    except Exception:
        return_dict['error'] = 'Annotation file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def load_predictions(prediction_file, return_dict):
    """Convert prediction JSON file."""

    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    id_set = set([])

    try:
        preds = json.load(open(prediction_file, 'r'))
    except Exception:
        return_dict['error'] = 'Prediction file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for pred in preds:
        if 'image_id' not in pred.keys():
            return_dict['warning'].append('There is an invalid annotation info, \
                likely missing key \'image_id\'.')
            continue
        if 'keypoint_annotations' not in pred.keys():
            return_dict['warning'].append(pred['image_id']+\
                ' does not have key \'keypoint_annotations\'.')
            continue
        image_id = pred['image_id'].split('.')[0]
        if image_id in id_set:
            return_dict['warning'].append(pred['image_id']+\
                ' is duplicated in prediction JSON file.')
        else:
            id_set.add(image_id)
        predictions['image_ids'].append(image_id)
        predictions['annos'][pred['image_id']] = dict()
        predictions['annos'][pred['image_id']]['keypoint_annos'] = pred['keypoint_annotations']

    return predictions


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = anno['keypoint_annos'].keys()[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = predict.keys()[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                    - predict_keypoints[visible, :2])**2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))
    return oks


def keypoint_eval(predictions, annotations, return_dict):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0
    
    # Construct set to speed up id searching.
    prediction_id_set = set(predictions['image_ids'])

    # for every annotation in our test/validation set
    for image_id in annotations['image_ids']:
        # if the image in the predictions, then compute oks
        if image_id in prediction_id_set:
            oks = compute_oks(anno=annotations['annos'][image_id], \
                              predict=predictions['annos'][image_id]['keypoint_annos'], \
                              delta=annotations['delta'])
            # view pairs with max OKSs as match ones, add to oks_all
            oks_all = np.concatenate((oks_all, np.max(oks, axis=0)), axis=0)
            # accumulate total num by max(gtN,pN)
            oks_num += np.max(oks.shape)
        else:
            # otherwise report warning
            return_dict['warning'].append(image_id+' is not in the prediction JSON file.')
            # number of humen in ground truth annotations
            gt_n = len(annotations['annos'][image_id]['human_annos'].keys())
            # fill 0 in oks scores
            oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
            # accumulate total num by ground truth number
            oks_num += gt_n

    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold)/np.float32(oks_num))
    return_dict['score'] = np.mean(average_precision)

    return return_dict


def main():
    """The evaluator."""

    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', help='prediction json file', type=str,
                        default='keypoint_predictions_example.json')
    parser.add_argument('--ref', help='annotation json file', type=str,
                        default='keypoint_annotations_example.json')
    args = parser.parse_args()

    # Initialize return_dict
    return_dict = dict()
    return_dict['error'] = None
    return_dict['warning'] = []
    return_dict['score'] = None

    # Load annotation JSON file
    start_time = time.time()
    annotations = load_annotations(anno_file=args.ref,
                                   return_dict=return_dict)
    print 'Complete reading annotation JSON file in %.2f seconds.' %(time.time() - start_time)

    # Load prediction JSON file
    start_time = time.time()
    predictions = load_predictions(prediction_file=args.submit,
                                   return_dict=return_dict)
    print 'Complete reading prediction JSON file in %.2f seconds.' %(time.time() - start_time)

    # Keypoint evaluation
    start_time = time.time()
    return_dict = keypoint_eval(predictions=predictions,
                                annotations=annotations,
                                return_dict=return_dict)
    print 'Complete evaluation in %.2f seconds.' %(time.time() - start_time)

    # Print return_dict and final score
    pprint.pprint(return_dict)
    print 'Score: ', '%.8f' % return_dict['score']


if __name__ == "__main__":

    main()
