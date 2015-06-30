#!/bin/bash

args=("$@")
datadir=${args[0]} # training data to be used for training
nbestlist=${args[1]} # nbest list to be used for testing
order=${args[2]} # order of the language model to be used
dumpdir=${args[3]} # path to dump the output of the reranker 
srcsgm=${args[4]} # path of the src to compare the reranker_output with
language=${args[5]} # langauge

./create-collection.x -i $datadir/* -c ../collections/reranking
./build-index.x -c ../collections/reranking/
./query-index-knm.x -c ../collections/reranking/ -p $nbestlist -m -n $order -b -r -path $dumpdir


perl wrap-xml.perl $language $srcsgm "SDSL" < output.rrank > output.sgm
perl multi-bleu.perl output.sgm< $srcsgm 







