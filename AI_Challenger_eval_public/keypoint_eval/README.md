# Human Skeleton System Keypoint Evaluation
Human Skeleton System Keypoint is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the final score (mAP) of the test result, based on your submited file and the reference file containing ground truth. 
# usage
```
python keypoint_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```   
A test case is provided, submited file is submit.json, reference file is ref.json, test it by:
```
python keypoint_eval.py --submit ./keypoint_predictions_example.json --ref ./keypoint_annotations_example.json   
```
The final score of the submited result, error message and warning message will be printed.
