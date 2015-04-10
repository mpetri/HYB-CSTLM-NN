#!/bin/bash
datadir="/home/ehsan/research/LM/SDSL/lm-sdsl/UnitTestData/data"

./create-collection.x -i $datadir/training.data -c ../collections/unittest
./build-index.x -c ../collections/unittest

for i in 2 3 4 5 6 7 8 9 10
do
    ./query-index-knm.x -c ../collections/unittest/ -p  $datadir/test.data -m false -n $i
done
