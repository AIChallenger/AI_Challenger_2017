#! /bin/sh

#Download the dataset and put the dataset in ../raw_data file

DATA_DIR=./t2t_data
TMP_DIR=../raw_data
mkdir -p $DATA_DIR

#unwrap xml for valid data and test data
python prepare_data/unwrap_xml.py $TMP_DIR/translation_validation_20170912/valid.en-zh.zh.sgm >$DATA_DIR/valid.en-zh.zh
python prepare_data/unwrap_xml.py $TMP_DIR/translation_validation_20170912/valid.en-zh.en.sgm >$DATA_DIR/valid.en-zh.en

#Prepare Data

##Chinese words segmentation
python prepare_data/jieba_cws.py $TMP_DIR/translation_train_20170912/train.zh > $DATA_DIR/train.zh
python prepare_data/jieba_cws.py $DATA_DIR/valid.en-zh.zh > $DATA_DIR/valid.zh
## Tokenize and Lowercase English training data
cat $TMP_DIR/translation_train_20170912/train.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/train.en
cat $DATA_DIR/valid.en-zh.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/valid.en

#Bulid Dictionary
python prepare_data/build_dictionary.py $DATA_DIR/train.en
python prepare_data/build_dictionary.py $DATA_DIR/train.zh
src_vocab_size=50000
trg_vocab_size=50000
python prepare_data/generate_vocab_from_json.py $DATA_DIR/train.en.json ${src_vocab_size} > $DATA_DIR/vocab.en
python prepare_data/generate_vocab_from_json.py $DATA_DIR/train.zh.json ${trg_vocab_size} > $DATA_DIR/vocab.zh
rm -r $DATA_DIR/train.*.json
