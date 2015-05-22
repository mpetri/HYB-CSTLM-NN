#!/bin/env python

import nltk


def lookup(tok, lexicon, add):
    i = lexicon.get(tok, None)
    if i != None: 
        return i
    elif add:
        i = len(lexicon)
        lexicon[tok] = i
        return i
    else:
        return lexicon['<UNK>']

def train(train_in_fname, train_out_fname):
    lexicon = dict()
    for i, term in enumerate('<EOF> <EOS> <UNK> <S> </S>'.split()):
        lexicon[term] = i
    
    tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    with open(train_in_fname) as fin:
        with open(train_out_fname, 'w') as fout:
            for line in fin:
                toks = tokenizer.tokenize(line)
                tids = map(lambda t: lookup(t, lexicon, True), toks)
                #globals().update(locals())
                for tok in tids:
                    print >>fout, tok,
                print >>fout
    
    return lexicon

def save_lexicon(lex_fname, lexicon):
    with open(lex_fname, 'w') as lout:
        for term, idx in sorted(lexicon.items()):
            print >>lout, term, idx

def load_lexicon(lex_fname):
    lexicon = dict()
    with open(lex_fname) as lin:
        for line in fin:
            term, idx = line.split(1)
            lexicon[term] = idx
    assert len(lexicon) >= 4
    assert max(lexicon.values()) == len(lexicon) - 1
    return lexicon

def test(test_in_fname, test_out_fname, lexicon):
    tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    with open(test_in_fname) as fin:
        with open(test_out_fname, 'w') as fout:
            for line in fin:
                toks = tokenizer.tokenize(line)
                tids = map(lambda t: lookup(t, lexicon, False), toks)
                for tok in tids:
                    print >>fout, tok,
                print >>fout

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--train-input', help='input training text file')
    parser.add_argument('-l', '--lexicon', help='lexicon file used as input or output')
    parser.add_argument('-o', '--train-output', help='output numberised file')
    parser.add_argument('-t', '--test-input', help='input testing text file')
    parser.add_argument('-u', '--test-output', help='output testing text file')
    args = parser.parse_args()

    def xor(a, b): return bool(a) ^ bool(b)

    if not args.train_input or not args.lexicon:
        parser.print_help()
        sys.exit()

    lex = None
    if args.train_input:
        print args.train_input, args.train_output
        lex = train(args.train_input, args.train_output)

    if args.lexicon:
        if lex != None:
            save_lexicon(args.lexicon, lex)
        else:
            lex = load_lexicon(args.lexicon)

    if args.test_input:
        assert args.test_output
        test(args.test_input, args.test_output, lex)

