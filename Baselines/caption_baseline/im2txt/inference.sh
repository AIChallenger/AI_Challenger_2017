#!/usr/bin/env bash

cd /your_work_dir
CHECKPOINT_PATH="/your_checkpoint_dir"
VOCAB_FILE="/your_word_list_dir/word_counts.txt"
export CUDA_VISIBLE_DEVICES="0"
IMAGE_DIR='/your_image_dir/caption_test1_images_20170923/'
OUTJSON='/your_output_dir/your_output.json'
python run_inference.py \
 	--checkpoint_path=${CHECKPOINT_PATH} \
  	--vocab_file=${VOCAB_FILE} \
  	--image_dir=${IMAGE_DIR}\
  	--out_predict_json=${OUTJSON}

