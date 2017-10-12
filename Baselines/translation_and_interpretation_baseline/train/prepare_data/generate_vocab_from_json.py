import json
import sys
import cPickle as pkl

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('usage: %s + input.json + vocab_size\n' % __file__)
        sys.exit(-1)
    vocab = load_dict(sys.argv[1])
    cnt = 0
    max_cnt = int(sys.argv[2])
    print('UNK')
    for k in vocab:
        if vocab[k] < max_cnt and k != '':
            if k == 'UNK':
                continue
            print k.strip()
