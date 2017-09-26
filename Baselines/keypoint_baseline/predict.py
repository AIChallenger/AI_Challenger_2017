import os
import argparse
from distutils.dir_util import mkpath
from utils import SSD, DeepLab, Keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default='data/valid/keypoint_validation_images_20170911')
parser.add_argument('--ssd_model', default='checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt')
parser.add_argument('--model', default='train/snapshots/model.ckpt-200000')
parser.add_argument('--out_dir', default='validation')
parser.add_argument('--out_file', default='predictions.json')
args = parser.parse_args()

mkpath(args.out_dir)

# temp files
crop_output_dir = os.path.join(args.out_dir, 'crop_images/')
SSD_output_json = os.path.join(args.out_dir, 'SSD_res.json')
crop_list = os.path.join(args.out_dir, 'crop_list.txt')
seg_output_dir = os.path.join(args.out_dir, 'crop_predictions/')


SSD.inference_for_humans(image_dir=args.img_dir,
                         checkpoint=args.ssd_model,
                         output_dir=crop_output_dir,
                         output_json=SSD_output_json,
                         net_shape=(512, 512))

# Create list file for cropped images,
with open(crop_list, 'w') as f:
    for img_file in os.listdir(crop_output_dir):
        f.write(img_file + '\n')


DeepLab.inference_for_keypoints(image_dir=crop_output_dir,
                                image_list=crop_list,
                                checkpoint=args.model,
                                output_dir=seg_output_dir)

Keypoints.make_json(SSD_json=SSD_output_json,
                    prediction_dir=seg_output_dir,
                    output_json=os.path.join(args.out_dir, args.out_file))
