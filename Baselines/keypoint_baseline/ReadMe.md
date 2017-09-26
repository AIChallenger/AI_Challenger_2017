### Description
This baseline method first uses SSD to locate each person appearing in the picture, then a semantic segmentation model is trained to identify each visible skeletal keypoint labeled by a rounded area within the bounding box of each person. 


### Setup the Environments
1. make sure python 2.7 and tensorflow are installed 
1. run `setup.sh`
1. download the dataset for the contest, put corresponding data files in `data/train`, `data/valid`, `data/test`
1. download [SSD Model](https://drive.google.com/file/d/0B0qPCUZ-3YwWT1RCLVZNN3RTVEU) and 
[Deeplab Model](https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU), 
put them in the `checkpoints` directory

### Train the Model
You can train the model using the following commands:
```bash
# preprocess the data
python prepro.py

# start training
python tensorflow-deeplab-resnet/train.py \
--data-dir=train/ \
--data-list=train/train.txt \
--snapshot-dir=train/snapshots/ \
--ignore-label=0 \
--is-training \
--not-restore-last \
--num-classes=15 \
--restore-from=checkpoints/deeplab_resnet.ckpt \
--batch-size=8 \
--num-steps=200000
```

Predictions on validation set can be generated using:
```bash
python predict.py --img_dir data/valid/keypoint_validation_images_20170911\
                  --out_dir validaton --out_file predictions.json \
                  --model train/snapshots/model.ckpt-200000
```

