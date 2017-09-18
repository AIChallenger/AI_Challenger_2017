#!/usr/bin/env bash
#outpath=/home/store-1-img/zhenghe/ai_challenger_caption_train_output/finetune.log
cd /home/zhenghe/caption_baseline/im2txt/im2txt
TFRECORD_DIR="/home/store-1-img/zhenghe/ai_challenger_caption_train_output"
INCEPTION_CHECKPOINT="/home/store-1-img/zhenghe/im2txt/data/inception_v3.ckpt"
MODEL_DIR="/home/store-1-img/zhenghe/ai_challenger_caption_train_output/model"
export CUDA_VISIBLE_DEVICES="1"
python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-00280" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=600 #> ${outpath} 2>&1 &