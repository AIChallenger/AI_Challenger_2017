# AI_Challenger
challenger.ai     
Scene classification baseline for AI_Challenger dataset.
# Requirements
- python 3.0 or greater
- TensorFlow 1.0 or greater
- opencv-python (3.2.0.7)
- Pillow (4.2.1)

# Prepare the Training Data
To train the model you will need to provide training data in numpy array format. We have provided a script (scene_input.py) to preprocess the AI_Challenger scene image dataset. The script will read the annotations file to get all the information. In each training step, it will fetch specific number of images according to the batch size, then resize each image, crop a square area and concate them in zero axis (form a larger batch as the final input data). The corresponding labels are also obtained. 

# Model Description
This simple model consists of three convolutional layers, three max pool layers and two fully connected layers. Local response normalization and dropout are also used. Details of network structure is in network.py.

# Training a Model
Run the training script.
```
python scene.py --mode train --train_dir TRAIN_IMAGE_PATH --annotations ANNOTATIONS_FILE_PATH --max_step 65000
```
The batch loss and accuracy will be logged in scene.log.
# Test a Model
Run the test script. 
```
python scene.py --mode test --test_dir TEST_IMAGE_PATH
```
Test result will be writed into JSON file named "submit.json", which contains image_id, top3 label_id as a list.
# Calculate Accuracy
Run the evaluation script. (This script is the lastest and robust version. In order to be compatible with label_id in string type, we always convert label_id to integer in reference data.)
```
python scene_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```
Top 3 accuracy in validation is about 40% with default training parameters.
# References
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.
