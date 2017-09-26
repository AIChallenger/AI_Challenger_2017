
# coding: utf-8

from __future__ import print_function
import os
import time
import warnings
from skimage import io
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, './tensorflow-deeplab-resnet/')
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label


IMG_MEAN = np.array((120.47387528, 113.77137482,
                     106.16930486), dtype=np.float32)  # Ours

NUM_CLASSES = 15  # 14 human keypoints and background
IGNORE_LABEL = 0  # background lable


# Main function.
def inference_for_keypoints(image_dir, image_list, checkpoint, output_dir):

    # Create directory for output.
    if not os.path.exists(output_dir):
        os.system('mkdir ' + output_dir)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            image_dir,
            image_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            IGNORE_LABEL,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label

    # Add one batch dimension.
    image_batch, label_batch = tf.expand_dims(
        image, dim=0), tf.expand_dims(label, dim=0)

    # Create network.
    net = DeepLabResNetModel({'data': image_batch},
                             is_training=False, num_classes=NUM_CLASSES)
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(
        raw_output, tf.shape(image_batch)[1:3, ])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3)  # Create 4-d tensor.

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    tf.train.Saver(var_list=restore_var).restore(sess, checkpoint)
    print('Successfully restored model parameters from {}'.format(checkpoint))

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Read list text file.
    with open(image_list, 'r') as txtfile:
        files = txtfile.readlines()

    # Iterate for all images.
    start = time.time()
    for idx, file in enumerate(files):

        # Print status.
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(files):
            print(str(idx + 1) + ' / ' + str(len(files)))

        # Inference for human keypoints, save the label image.
        preds = sess.run(pred)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_dir + '/' + file.split('.')
                      [0] + '.png', np.uint8(np.squeeze(preds)))

    print('Successfully processed all the images in %.2f seconds.' %
          (time.time() - start))

    coord.request_stop()
    coord.join(threads)
