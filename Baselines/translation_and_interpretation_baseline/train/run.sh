bin=../tensor2tensor/bin
python $bin/t2t-trainer --registry_help

export CUDA_VISIBLE_DEVICES=4,5,6,7
PROBLEM=translate_enzh
MODEL=transformer
HPARAMS=transformer_big
HOME=`pwd`
DATA_DIR=$HOME/t2t_data
TMP_DIR=$DATA_DIR
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
python $bin/t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --num_shards=100 \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=2048' or even 1024.
python $bin/t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=2048,shared_embedding_and_softmax_weights=0'
