#!/bin/sh

# demonstrate a bug

rm -r ../collections/undoc
python numberise.py -i ../UnitTestData/data/undoc_2000_fr_en_sample.train.txt -l ../UnitTestData/data/undoc_2000_fr_en_sample.lex -o ../UnitTestData/data/undoc_2000_fr_en_sample.train -t ../UnitTestData/data/undoc_2000_fr_en_sample.test.txt -u ../UnitTestData/data/undoc_2000_fr_en_sample.test
./create-collection.x -c ../collections/undoc -i ../UnitTestData/data/undoc_2000_fr_en_sample.train
touch ../collections/undoc/text.VOCAB
./build-index.x -c ../collections/undoc

# these calls return a different perplexity (the -b is right, to my eye) -- comes down to ngrams containing UNK token "2"
./query-index-knm.x -c ../collections/undoc -p ../UnitTestData/data/undoc_2000_fr_en_sample.test -n 5
./query-index-knm.x -c ../collections/undoc -p ../UnitTestData/data/undoc_2000_fr_en_sample.test -n 5 -b
