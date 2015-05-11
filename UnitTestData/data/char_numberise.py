#!/bin/env python

import sys, nltk

# word tokenizer
#tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()

lexicon = dict()
for i, term in enumerate('EOF EOS UNK <s> </s>'.split()):
    lexicon[term] = i

def lookup(tok, lexicon=lexicon):
    i = lexicon.get(tok, None)
    if i != None: return i
    i = len(lexicon)
    lexicon[tok] = i
    return i

with open(sys.argv[1]) as fin:
    for line in fin:
        # toks = tokenizer.tokenize(line)
        toks = line.strip() # just use the character sequence
        tids = map(lookup, toks)
        for tok in tids:
            print tok,
        print

if len(sys.argv) >= 3:
    with open(sys.argv[2], 'w') as lout:
        for term, idx in sorted(lexicon.items()):
            print >>lout, term, idx
