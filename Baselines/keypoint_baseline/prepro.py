from __future__ import print_function
import os
import json
import time
import warnings
import argparse
import numpy as np
from distutils.dir_util import mkpath
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/train')
parser.add_argument('--out_dir', default='train')
parser.add_argument('--ratio', type=float, default=2.)
parser.add_argument('--std', type=float, default=0.02)
args = parser.parse_args()

image_dir = os.path.join(args.data_dir, 'keypoint_train_images_20170902/')
images = os.listdir(image_dir)
print('images', len(images))

json_file = os.path.join(args.data_dir, 'keypoint_train_annotations_20170909.json')
annos = json.load(open(json_file, 'r'))
print('annos', len(annos))


target_image_dir = os.path.join(args.out_dir, 'human_images/')
target_mask_dir = os.path.join(args.out_dir, 'human_masks/')
mkpath(target_image_dir)
mkpath(target_mask_dir)

keypoints_std = np.ones(14) * args.std

start = time.time()
file_mapping = []
for idx, anno in enumerate(annos):

    # Print status.
    if (idx + 1) % 1000 == 0 or (idx + 1) == len(annos):
        print(str(idx + 1) + ' / ' + str(len(annos)))

    try:
        # Read image.
        img = io.imread(image_dir + anno['image_id'] + '.jpg')
        height, width, channel = img.shape

        # For every human annotations.
        for key in anno['keypoint_annotations'].keys():

            # Read keypoint positions and the bounding box
            keypoints = np.reshape(anno['keypoint_annotations'][key], (14, 3))
            bbox = anno['human_annotations'][key]
            mask = np.zeros((height, width), 'uint8')

            # Expand bounding box by 10%.
            expand_x = int(0.1 * np.float32(bbox[2] - bbox[0]))
            expand_y = int(0.1 * np.float32(bbox[3] - bbox[1]))
            scale = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

            crop_left = np.max((bbox[0] - expand_x, 0))
            crop_right = np.min((bbox[2] + expand_x, width - 1))
            crop_up = np.max((bbox[1] - expand_y, 0))
            crop_down = np.min((bbox[3] + expand_y, height - 1))

            # For every visible keypoint.
            for i in range(14):
                if keypoints[i, 2] == 1:
                    # Center and radius of the keypoint.
                    c = keypoints[i, :2]
                    r = int(keypoints_std[i] * args.ratio * scale) + 1
                    # Boundary of each keypoint area.
                    left = max((0, c[0] - r))
                    right = min((width - 1, c[0] + r))
                    up = max((0, c[1] - r))
                    down = min((height - 1, c[1] + r))
                    # Generate mask of each keypoint area.
                    meshgrid = np.meshgrid(range(left, right), range(up, down))
                    inds = ((meshgrid[0] - c[0]) ** 2 + (meshgrid[1] - c[1]) ** 2) <= r * r
                    mask[up:down, left:right][inds] = i + 1

            # Crop the original image and the ground truth mask.
            img_crop = img[crop_up:crop_down, crop_left:crop_right, :]
            mask_crop = mask[crop_up:crop_down, crop_left:crop_right]

            # Suppress the warning of saving low contrast images.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Save the cropped images and cropped ground truth mask
                image_file = os.path.join(target_image_dir, anno['image_id'] + '_' + key + '.jpg')
                mask_file = os.path.join(target_mask_dir, anno['image_id'] + '_' + key + '.png')
                io.imsave(image_file, img_crop)
                io.imsave(mask_file, mask_crop)
                file_mapping.append((image_file, mask_file))
    except:
        continue

with open(os.path.join(args.out_dir, 'train.txt'), 'w') as f:
    for image_file, mask_file in file_mapping:
        f.write('{} {}\n'.format(image_file, mask_file))

print('Successfully processed all the images in %.2f seconds.' % (time.time() - start))
