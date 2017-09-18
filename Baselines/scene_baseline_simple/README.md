# AI_Challenger
challenger.ai     
Scene classification baseline for AI_Challenger dataset.
# Requirements
- TensorFlow 1.0 or greater
- opencv-python (3.2.0.7)
- Pillow (4.2.1)

# Prepare the Training Data
To train the model you will need to provide training data in numpy array format. We have provided a script (scene_input.py) to preprocess the AI_Challenger scene image dataset. The script will read the annotations file to get all the information. In each training step, it will fetch specific number of images according to the batch size, resize each images and crop a square area, concate them in zero axis into a larger batch as the final input data. The corresponding labels are also obtained. 

# Model description
This simple model consists of three convolutional layers, three max pool layers and two fully connected layers. Local response normalization and dropout are also used. Details of network structure is in network.py.

# Training a Model
Run the training script.
```
python scene.py --mode train --train_dir train_images_path --annotations annotations_file_path --max_step 65000
```
The batch loss and accuracy will be logged in scene.log.
# Test a Model
Run the test script. 
```
python scene.py --mode test --test_dir test_images_path
```
Test result will be writed into JSON file named "submit.json", which contains image_id, top3 label_id as a list.
# References
Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.