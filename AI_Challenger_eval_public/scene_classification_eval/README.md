# Scene Classification Evaluation
Scene classification is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the accuracy of the test result, based on your submited file and the reference file containing ground truth. 
# Usage
```
python scene_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```   
A test case is provided, submited file is submit.json, reference file is ref.json, test it by:
```
python scene_eval.py --submit ./submit.json --ref ./ref.json   
```
The accuracy of the submited result, error message and warning message will be printed.    