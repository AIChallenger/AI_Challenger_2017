## Requirements ##
- java 1.8.0
- python 2.7
- jieba 0.38

jieba 采用"精确模式" jieba.cut(～, cut_all=False)
```
cd caption-eval/
```
**Prepare for your own reference data**    
`import hashlib` is needed.     
`your_reference.json` should be like `data/id_to_words.json`, and `file_name` is the image name, id can be computed by using `image_hash = int(int(hashlib.sha256(img_name).hexdigest(), 16) % sys.maxint)`.    
`your_submit_data.json` should be like `data/id_to_test_caption.json`

**Evaluate the model predictions against the references**    
Sample file with predictions from a model is in `data/id_to_test_caption.json`
Sample file with reference is in `data/id_to_words.json`     
json_predictions_file = "your_result_json_file"     
reference_file = "reference_json_file"     
For test you can just run     
```
python run_evaluations.py
```
For your own test data, you can run

```
python run_evaluations.py -submit your_submit_data.json -ref your_reference.json
```

## References ##

- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325) Changes have been made to support Chinese contents.
- http://ms-multimedia-challenge.com/2017/leaderboard
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation] (http://arxiv.org/pdf/1411.5726.pdf)
- jieba: We use the “精确模式” jieba.cut(～, cut_all=False）which is included in https://github.com/fxsjy/jieba
- https://github.com/vsubhashini/caption-eval.git

## Developers ##
- He Zheng
- Jiahong Wu
- Chuang Zhou
- Yixin Li
- Baoming Yan


## Acknowledgement ##
- Rui Liang
- Yonggang Wang

