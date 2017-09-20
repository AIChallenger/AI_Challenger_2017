#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
'''
# AI_Challenger
Scene classification baseline for AI_Challenger dataset.
# Requirements
- TensorFlow 1.0 or greater
- opencv-python (3.2.0.7)
- Pillow (4.2.1)

# Prepare the Training Data
To train the model you will need to provide training data in numpy array format. 
We have provided a script (scene_input.py) to preprocess the AI_Challenger scene image dataset. 
The script will read the annotations file to get all the information. 
In each training step, it will fetch specific number of images according to the batch size, 
resize each images and crop a square area, concate them in zero axis into 
a larger batch as the final input data. The corresponding labels are also obtained. 

# Model description
This simple model consists of three convolutional layers, 
three max pool layers and two fully connected layers. 
Local response normalization and dropout are also used. 
Details of network structure is in network.py.

# Training a Model
Run the training script.
```
python scene.py --mode train --train_dir train_images_path --annotations annotations_file_path --max_step 65000
```
The batch loss and accuracy will be logged in scene.log.
# Test a Model
Run the test script. 
```
python scene.py --mode test --test_dir test_images_path
```
Test result will be writed into JSON file named "submit.json", which contains image_id, top3 label_id as a list.
# References
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]
//Advances in neural information processing systems. 2012: 1097-1105.
'''
import tensorflow as tf
import numpy as np
import time
import argparse
import cv2
import json
import os
import scene_input
import network

BATCH_SIZE = 32
IMAGE_SIZE = 128
IMAGE_CHANNEL = 3
CHECKFILE = './checkpoint/model.ckpt'
LOGNAME = 'scene'


def train(train_dir, annotations, max_step, checkpoint_dir='./checkpoint/'):
    # train the model
    scene_data = scene_input.scene_data_fn(train_dir, annotations)
    features = tf.placeholder("float32", shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name="features")
    labels = tf.placeholder("float32", [None], name="labels")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=80)
    train_step, cross_entropy, logits, keep_prob = network.inference(features, one_hot_labels)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')
        logger = scene_input.train_log(LOGNAME)

        for step in range(start_step, start_step + max_step):
            start_time = time.time()
            x, y = scene_data.next_batch(BATCH_SIZE, IMAGE_SIZE)
            sess.run(train_step, feed_dict={features: x, labels: y, keep_prob: 0.5})
            if step % 50 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: x, labels: y, keep_prob: 1})
                train_loss = sess.run(cross_entropy, feed_dict={features: x, labels: y, keep_prob: 1})
                duration = time.time() - start_time
                logger.info("step %d: training accuracy %g, loss is %g (%0.3f sec)" % (step, train_accuracy, train_loss, duration))
            if step % 1000 == 1:
                saver.save(sess, CHECKFILE, global_step=step)
                print('writing checkpoint at step %s' % step)


def test(test_dir, checkpoint_dir='./checkpoint/'):
    # predict the result 
    test_images = os.listdir(test_dir)
    features = tf.placeholder("float32", shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name="features")
    labels = tf.placeholder("float32", [None], name="labels")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=80)
    train_step, cross_entropy, logits, keep_prob = network.inference(features, one_hot_labels)
    values, indices = tf.nn.top_k(logits, 3)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            raise Exception('no checkpoint find')

        result = []
        for test_image in test_images:
            temp_dict = {}
            x = scene_input.img_resize(os.path.join(test_dir, test_image), IMAGE_SIZE)
            predictions = np.squeeze(sess.run(indices, feed_dict={features: np.expand_dims(x, axis=0), keep_prob: 1}), axis=0)
            temp_dict['image_id'] = test_image
            temp_dict['label_id'] = predictions.tolist()
            result.append(temp_dict)
            print('image %s is %d,%d,%d' % (test_image, predictions[0], predictions[1], predictions[2]))
        
        with open('submit.json', 'w') as f:
            json.dump(result, f)
            print('write result json, num is %d' % len(result))
       

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help="""\
        determine train or test\
        """
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='../ai_challenger_scene_train_20170904/scene_train_images_20170904/',
        help="""\
        determine path of trian images\
        """
    )

    parser.add_argument(
        '--annotations',
        type=str,
        default='../ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json',
        help="""\
        annotations for train images\
        """
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../ai_challenger_scene_validation_20170908/scene_validation_images_20170908/',
        help="""\
        determine path of test images\
        """
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=65000,
        help="""\
        determine maximum training step\
        """
    )

    FLAGS = parser.parse_args()
    if FLAGS.mode == 'train':
        train(FLAGS.train_dir, FLAGS. annotations, FLAGS.max_step)
    elif FLAGS.mode == 'test':
        test(FLAGS.test_dir)
    else:
        raise Exception('error mode')
    print('done')
