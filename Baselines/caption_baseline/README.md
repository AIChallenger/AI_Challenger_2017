# AI_Challenger
challenger.ai     
Image caption (Chinese) baseline for AI_Challenger dataset.
# Requirements
- python 2.7
- TensorFlow 1.0 or greater
- jieba 0.38 

# Prepare the Training Data
To train the model you will need to provide training data in native TFRecord format. The TFRecord format consists of a set of sharded files containing serialized tf.SequenceExample protocol buffers. Each tf.SequenceExample proto contains an image (JPEG format), a caption and metadata such as the image id.

Each caption is a list of words. During preprocessing, a dictionary is created that assigns each word in the vocabulary to an integer-valued id. Each caption is encoded as a list of integer word ids in the tf.SequenceExample protos.

We have provided a script to preprocess the AI_Challenger image captioning data set into this format. preprocessing the data may take several hours depending on your network and computer speed. Please be patient.
```
unzip ai_challenger_caption_train_20170902.zip
chmod +x build_tfrecord.sh
./build_tfrecord.sh
```
You can change `image_dir`, `captions_file`, `output_dir`, `train_shards`, `num_threads` in build_tfrecord.sh.    
```
image_dir=your_data_dir/ai_challenger_caption_train_20170902/caption_train_images_20170902
captions_file=yout_data_dir/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json     
```

When the script finishes you will find 280 training files in `output_dir`. The files will match the patterns train-?????-of-00280.    
More details can be found at References.     

# Download the Inception v3 Checkpoint
Location to save the Inception v3 checkpoint.
```
INCEPTION_DIR="${HOME}/im2txt/data"
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"
```
# Training a Model
## Initial Training
Run the training script.
```
./train.sh
```
Note that you should change data_dir in train.sh
## Fine Tune the Inception v3 Model
Run the training script.
```
./finetune.sh
```
Note that you should change data_dir in finetune.sh
# References

Show and Tell: A Neural Image Caption Generator

Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.

IEEE transactions on pattern analysis and machine intelligence (2016).

Full text available at: http://arxiv.org/abs/1609.06647    
Code availabel at: https://github.com/tensorflow/models/tree/master/im2txt