# coding: utf-8

import sys
sys.path.append('./SSD-Tensorflow/')

import os
import math
import random
import json
import time

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
from skimage import io

from nets import ssd_vgg_300, ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing


# TensorFlow session: grow memory when needed.
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.InteractiveSession(config=config)


# Load SSD net and return tensors using for inference.
def load_ssd_net(checkpoint, net_shape=(300, 300), data_format='NHWC'):

    # Input placeholder.
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = \
        ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format,
            resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net_dict = {300: ssd_vgg_300.SSDNet, 512: ssd_vgg_512.SSDNet}
    ssd_net = ssd_net_dict[net_shape[0]]()

    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(
            image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # Return dictionary.
    return_dict = {'img_input': img_input,
                   'image_4d': image_4d,
                   'predictions': predictions,
                   'localisations': localisations,
                   'bbox_img': bbox_img,
                   'ssd_anchors': ssd_anchors}
    return return_dict


# Main image processing routine.
def process_image(img, tensors_dict, select_threshold=0.5, nms_threshold=.45,
                  net_shape=(300, 300)):

    # Read tensors in tensors_dict.
    img_input = tensors_dict['img_input']
    image_4d = tensors_dict['image_4d']
    predictions = tensors_dict['predictions']
    localisations = tensors_dict['localisations']
    bbox_img = tensors_dict['bbox_img']
    ssd_anchors = tensors_dict['ssd_anchors']

    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img \
        = sess.run([image_4d, predictions, localisations, bbox_img],
                   feed_dict={img_input: img})
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21,
        decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(
        rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(
        rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Save croped images.
def save_crop_images(img, detection_result, output_dir, filename, thresh=0.5):

    scores = np.array(detection_result['scores'])
    bboxes = np.array(detection_result['bboxes'])
    move = []

    for i in range(len(scores)):
        if scores[i] >= thresh:

            # Expand boundary by 1/10.
            bbox = bboxes[i, :]
            expand_x = int(
                0.1 * np.float32(img.shape[1]) * (bbox[3] - bbox[1]))
            expand_y = int(
                0.1 * np.float32(img.shape[0]) * (bbox[2] - bbox[0]))

            # Define boundary edges.
            left = np.max((int(bbox[1] * img.shape[1]) - expand_x, 0))
            right = np.min(
                (int(bbox[3] * img.shape[1]) + expand_x, img.shape[1] - 1))
            up = np.max((int(bbox[0] * img.shape[0]) - expand_y, 0))
            down = np.min(
                (int(bbox[2] * img.shape[0]) + expand_y, img.shape[0] - 1))

            ratio = np.float32((right - left) * (down - up)) / \
                np.float32(img.shape[0] * img.shape[1])
            move.append([left, up])

            # Crop human from the original image when its large enough.
            if ratio > 0.03:
                img_crop = img[up:down, left:right, :]
            else:
                continue

            # Save croped image.
            file_path = output_dir + \
                filename.split('.')[0] + '_' + str(i) + '.jpg'
            io.imsave(file_path, img_crop)

    return move

# Main function: SSD inference for humans.
def inference_for_humans(image_dir, checkpoint, output_dir, output_json,
                         net_shape=(512, 512), thresh=0.5):

    # Create directory for output.
    if not os.path.exists(output_dir):
        os.system('mkdir ' + output_dir)

    # Load SSD net structure.
    start = time.time()
    tensors_dict = load_ssd_net(checkpoint=checkpoint, net_shape=net_shape)
    print('Successfully loaded SSD net in %.2f seconds.' %
          (time.time() - start))

    # List all the images
    files = sorted(os.listdir(image_dir))
    detection_results = dict()
    start = time.time()

    for idx, file in enumerate(files):

        # Print status.
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(files):
            print(str(idx + 1) + ' / ' + str(len(files)))

        # Read image, feed into SSD
        img = io.imread(image_dir + '/' + file)
        rclasses, rscores, rbboxes = process_image(
            img, net_shape=net_shape, tensors_dict=tensors_dict)
        # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

        # Find human detection results
        detection_results[file] = dict()
        detection_results[file]['scores'] = rscores[rclasses == 15].tolist()
        detection_results[file]['bboxes'] = rbboxes[rclasses == 15, :].tolist()

        # Save Croped images
        move = save_crop_images(img=img,
                                detection_result=detection_results[file],
                                output_dir=output_dir,
                                filename=file,
                                thresh=0.5)
        detection_results[file]['move'] = move

    # Dump detection result JSON file and create listing text file.
    json.dump(detection_results, open(output_json, 'w'))

    # Reset default graph for next procedure.
    tf.reset_default_graph()
    print('Successfully processed all the images in %.2f seconds.' %
          (time.time() - start))
