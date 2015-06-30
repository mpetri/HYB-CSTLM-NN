#!/bin/bash

args=("$@")
datadir=${args[0]} # training data to be used for training
nbestlist=${args[1]} # nbest list to be used for testing
order=${args[2]} # order of the language model to be used
srcsgm=${args[3]} # path of the src to compare the reranker_output with
language=${args[4]} # langauge: en, fr, cz, etc

rm -rf ../collections/reranking
./create-collection.x -i $datadir/* -c ../collections/reranking
./build-index.x -c ../collections/reranking/
./query-index-knm.x -c ../collections/reranking/ -p $nbestlist -m -n $order -b -r








