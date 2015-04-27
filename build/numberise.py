#!/bin/env python

import sys, nltk

tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()


lexicon = dict()
lexicon['sentinel 0'] = 0
lexicon['sentinel 1'] = 1
lexicon['sentinel 2'] = 2
lexicon['sentinel 3'] = 3

def lookup(tok, lexicon=lexicon):
    i = lexicon.get(tok, None)
    if i != None: return i
    i = len(lexicon)
    lexicon[tok] = i
    return i

with open(sys.argv[1]) as fin:
    for line in fin:
        toks = tokenizer.tokenize(line)
        tids = map(lookup, toks)
        for tok in tids:
            print tok,
        print
