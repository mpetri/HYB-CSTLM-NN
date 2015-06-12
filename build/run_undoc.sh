#!/bin/sh

trn=../UnitTestData/data/undoc_2000_fr_en_100k.train
tst=../UnitTestData/data/undoc_2000_fr_en_100k.test

rm -rf ../collections/undoc
./create-collection.x -c ../collections/undoc -i $trn
./build-index.x -c ../collections/undoc -d -m

echo "==================== KN ===================="
./query-index-knm.x -c ../collections/undoc -p $tst -n 5 -s | tee out.succinct

echo "==================== MKN ===================="
./query-index-knm.x -c ../collections/undoc -p $tst -n 5 -s -m | tee out.succinct

#echo
#echo "==================== index succinct compute n1fb ===================="
#./query-index-knm.x -c ../collections/undoc -p $tst -n 5 -b | tee out.compute
