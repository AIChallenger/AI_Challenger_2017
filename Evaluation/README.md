## Files ##
./caption_eval
- Evaluation utility for image Chinese captioning task.

./keypoint_eval
- Evaluation utility for human skeleton system keypoint task.

./scene_classification_eval
- Evaluation utility for scene classification task.

./interpretation_and_translation_eval
- Evaluation utility for interpretation and translation task.


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
