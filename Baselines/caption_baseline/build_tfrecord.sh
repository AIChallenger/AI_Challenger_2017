#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="1"
python build_tfrecord.py --image_dir=/home/store-1-img/wenjia/challenger_dataset/caption/ai_challenger_caption_train_20170902/caption_train_images_20170902\
                        --captions_file=/home/store-1-img/wenjia/challenger_dataset/caption/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json\
                        --output_dir=/home/store-1-img/zhenghe/ai_challenger_caption_train_output\
                        --train_shards=280\
                        --num_threads=56
