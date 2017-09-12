# encoding: utf-8
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
"""The unittest for image Chinese captioning evaluation."""
# __author__ = 'ZhengHe'
# python2.7
# python run_evaluations.py

import sys
import unittest

reload(sys)
sys.setdefaultencoding('utf8')
from run_evaluations import compute_m1


class TestComputem1(unittest.TestCase):
    """class for test"""

    def test_rightdata(self):
        """test for right data"""
        m1_score = compute_m1(json_predictions_file="data/id_to_test_caption.json",
                              reference_file="data/id_to_words.json")
        self.assertEqual(m1_score['error'], 0)

    def test_nulldata(self):
        """test for null data"""
        m1_score = compute_m1(json_predictions_file="data/has_null_data.json",
                              reference_file="data/id_to_words.json")
        self.assertEqual(m1_score['error'], 1)

    def test_keyerror(self):
        """test for key error"""
        m1_score = compute_m1(json_predictions_file="data/key_error.json",
                              reference_file="data/id_to_words.json")
        self.assertEqual(m1_score['error'], 1)

    def test_lessdatanumber(self):
        """test for less data"""
        m1_score = compute_m1(json_predictions_file="data/less_data_number.json",
                              reference_file="data/id_to_words.json")
        self.assertEqual(m1_score['error'], 1)

    def test_moredatanumber(self):
        """test for more data"""
        m1_score = compute_m1(json_predictions_file="data/more_data_number.json",
                              reference_file="data/id_to_words.json")
        self.assertEqual(m1_score['error'], 1)

    def test_wrongname(self):
        """test for wrong_name"""
        m1_score = compute_m1(json_predictions_file="data/wrong_name.json",
                              reference_file="data/id_to_words.json")
        self.assertEqual(m1_score['error'], 1)


if __name__ == "__main__":
    unittest.main()
