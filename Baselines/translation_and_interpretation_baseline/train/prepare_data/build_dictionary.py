#!/bin/env python

import numpy
import json

import sys
import fileinput
import argparse
from collections import OrderedDict

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("file", help="input files")
    p.add_argument('-t','--tokens', nargs='+', default=[], help="special tokens")
    p.add_argument('-d','--dict', type=str, help="User defined dict")
    return p.parse_args()

def load_user_dict(file, user_dict):
    fh=open(file, 'r')
    for line in fh:
        if not line:
            break
        if line == '':
            continue
        line=line.rstrip()
        words = line.split(' ')
        for word in words:
            user_dict.add(word)

def main():
    args = parse_args()
    # open file
    user_dict = set()
    if args.dict:
        load_user_dict(args.dict, user_dict)
    if True:
        filename=args.file
        print 'Processing', filename
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = word_freqs.keys()
        freqs = word_freqs.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['eos'] = 0
        worddict['UNK'] = 1

        # add user defined tokens
        ii = 2
        tokens = ['eos', 'UNK']
        for ww in args.tokens:
            worddict[ww] = ii
            tokens += [ww]
            ii += 1
        # add user defined dictiona
        for ww in user_dict:
            if ww in tokens:
                continue
            worddict[ww] = ii
            tokens += [ww]
            ii += 1
        # add words from corpus
        for ww in sorted_words:
            if ww in tokens:
                continue
            worddict[ww] = ii
            ii += 1

        with open('%s.json'%filename, 'wb') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        print 'Done'

if __name__ == '__main__':
    main()
