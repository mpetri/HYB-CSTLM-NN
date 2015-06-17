#!/bin/sh

trn=../UnitTestData/data/undoc_2000_fr_en_sample.txt
tst=../UnitTestData/data/undoc_2000_fr_en_sample2.txt

rm -rf ../collections/undoc
./create-collection.x -c ../collections/undoc -i $trn -1
./build-index.x -c ../collections/undoc -d -m

echo "==================== KN ===================="
./query-index-knm.x -c ../collections/undoc -p $tst -n 9999 -s -1 

echo "==================== MKN ===================="
./query-index-knm.x -c ../collections/undoc -p $tst -n 9999 -s -m -1 
