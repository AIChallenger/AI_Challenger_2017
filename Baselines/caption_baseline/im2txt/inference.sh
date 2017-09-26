#!/usr/bin/env bash

cd /home/zhenghe/caption_baseline/im2txt/im2txt
CHECKPOINT_PATH="/home/store-1-img/zhenghe/ai_challenger_caption_train_output/model/train"
VOCAB_FILE="/home/store-1-img/zhenghe/ai_challenger_caption_train_output/word_counts.txt"
export CUDA_VISIBLE_DEVICES="0"
IMAGE_DIR='/home/store-1-img/zhenghe/caption_faceplusplus/wujiahong/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/'
OUTJSON='/home/store-1-img/zhenghe/test1_predict_20170925.json'
python run_inference.py \
 	--checkpoint_path=${CHECKPOINT_PATH} \
  	--vocab_file=${VOCAB_FILE} \
  	--image_dir=${IMAGE_DIR}\
  	--out_predict_json=${OUTJSON}

