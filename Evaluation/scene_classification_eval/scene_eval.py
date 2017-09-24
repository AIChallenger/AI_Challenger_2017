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

"""
Scene classification is a task of AI Challenger 全球AI挑战赛

This python script is used for calculating the accuracy of the test result,

based on your submited file and the reference file containing ground truth.

Usage:

python scene_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH

A test case is provided, submited file is submit.json, reference file is ref.json, test it by:

python scene_eval.py --submit ./submit.json --ref ./ref.json

The accuracy of the submited result, error message and warning message will be printed.
"""

import json
import argparse
import time


def __load_data(submit_file, reference_file):
  # load submit result and reference result

    with open(submit_file, 'r') as file1:
        submit_data = json.load(file1)
    with open(reference_file, 'r') as file1:
        ref_data = json.load(file1)
    if len(submit_data) != len(ref_data):
        result['warning'].append('Inconsistent number of images between submission and reference data \n')
    submit_dict = {}
    ref_dict = {}
    for item in submit_data:
        submit_dict[item['image_id']] = item['label_id']
    for item in ref_data:
        ref_dict[item['image_id']] = int(item['label_id'])
    return submit_dict, ref_dict


def __eval_result(submit_dict, ref_dict):
    # eval accuracy

    right_count = 0
    for (key, value) in ref_dict.items():

        if key not in set(submit_dict.keys()):
            result['warning'].append('lacking image %s in your submission file \n' % key)
            print('warnning: lacking image %s in your submission file' % key)
            continue

        if value in submit_dict[key][:3]:
            right_count += 1

    result['score'] = str(float(right_count)/max(len(ref_dict), 1e-5))
    return result


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--submit',
        type=str,
        default='./submit.json',
        help="""\
        Path to submission file\
        """
    )

    PARSER.add_argument(
        '--ref',
        type=str,
        default='./ref.json',
        help="""\
        Path to reference file\
        """
    )

    FLAGS = PARSER.parse_args()

    result = {'error': [], 'warning': [], 'score': None}

    START_TIME = time.time()
    SUBMIT = {}
    REF = {}

    try:
        SUBMIT, REF = __load_data(FLAGS.submit, FLAGS.ref)
    except Exception as error:
        result['error'].append(str(error))
    try:
        result = __eval_result(SUBMIT, REF)
    except Exception as error:
        result['error'].append(str(error))
    print('Evaluation time of your result: %f s' % (time.time() - START_TIME))

    print(result)
