#%matplotlib inline
from pycxtools.coco import COCO
from pycxevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
# set up file names and pathes
dataDir='.'
dataDir1='/media/zh/E/im2txt/data/mscoco/raw-data'
dataType='val2014'
algName = 'fakecap'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]
print('annFile : %s'% annFile)
# create coco object and cocoRes object

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()
# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))
# demo how to use evalImgs to retrieve low score result
print(len(cocoEval.evalImgs))
exit()
evals = [eva for eva in cocoEval.evalImgs if eva['CIDEr']<30]
print('ground truth captions')
imgId = evals[0]['image_id']
annIds = coco.getAnnIds(imgIds=imgId)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

print('\n')
print('generated caption (CIDEr score %0.1f)'%(evals[0]['CIDEr']))
annIds = cocoRes.getAnnIds(imgIds=imgId)
anns = cocoRes.loadAnns(annIds)
coco.showAnns(anns)

# img = coco.loadImgs(imgId)[0]
# I = io.imread('%s/%s/%s'%(dataDir1,dataType,img['file_name']))
# plt.figure(1)
# plt.imshow(I)
# plt.axis('off')
# plt.show()
# # plot score histogram
# ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]
# plt.figure(2)
# plt.hist(ciderScores)
# plt.title('Histogram of CIDEr Scores', fontsize=20)
# plt.xlabel('CIDEr score', fontsize=20)
# plt.ylabel('result counts', fontsize=20)
# plt.show()
# # save evaluation results to ./results folder
# json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
# json.dump(cocoEval.eval,     open(evalFile, 'w'))
