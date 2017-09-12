# Human Skeletal System Keypoint Evaluation
Human Skeletal System Keypoint is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the submission score, mean Average Precision(mAP), based on your submission file and the reference file containing ground truth. 
# Usage
```
python keypoint_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```   
A test case is provided, where keypoint_predictions_example.json is the submission file, and keypoint_annotations_example.json is the reference file. You can run the test by:
```
python keypoint_eval.py --submit ./keypoint_predictions_example.json --ref ./keypoint_annotations_example.json   
```
The final score of the submission result, error messages and warning messages will be printed.
