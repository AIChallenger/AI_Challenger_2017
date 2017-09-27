# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import io


import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "/your_checkpoint_path",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/your_word_dir/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("image_dir", "/your_image_dir/image_test/",
                       "image dir of images for test "
                       "of image files.")
tf.flags.DEFINE_string("out_predict_json", "/your_json_dir/your_output.json",
                       "Out put predict json file")


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  dirs = os.walk(FLAGS.image_dir)
  for a, _, filelist in dirs:
    for filename in filelist:
      origin_name = a  + filename
      if origin_name.endswith('.jpg'):
        filenames.append(origin_name)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)


    res = []
    num = 1
    for filename in filenames:
      imgid_sentence = {}
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      # print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = "".join(sentence)
        if i == 0:
          if num % 100 ==0 :
            print("Captions for image %s:" % os.path.basename(filename))
            print("%d) %s (p=%f)" % (num,sentence, math.exp(caption.logprob)))
          imgid_sentence['image_id'] = os.path.basename(filename).split('.')[0]
          imgid_sentence['caption'] = sentence
          res.append(imgid_sentence)
      num = num + 1

    with io.open(FLAGS.out_predict_json, 'w', encoding='utf-8') as fd:
      fd.write(unicode(json.dumps(res,
                                  ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': '))))
    assert len(filenames) == len(res)
    print("Finished process %d images!"%len(filenames))

if __name__ == "__main__":
  tf.app.run()
