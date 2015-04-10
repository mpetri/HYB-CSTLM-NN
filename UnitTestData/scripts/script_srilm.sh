#!/bin/bash
datadir="/home/ehsan/research/LM/SDSL/lm-sdsl/UnitTestData/data"
experimentdir="/home/ehsan/research/LM/experiments/UnitTestData"
mkdir $experimentdir
for i in 2 3 4 5 6 7 8 9 10
do
    param=""
    if (($i==2))
    then
		param="-gt1min 0 -gt2min 0"
    elif (($i==3))
    then
        param="-gt1min 0 -gt2min 0 -gt3min 0"
    elif (($i==4))
    then
        param="-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0"
    elif (($i==5))
    then
        param="-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0"
    elif (($i==6))
    then
        param="-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0 -gt6min 0"
    elif (($i==7))
    then
        param="-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0 -gt6min 0 -gt7min 0"
    elif (($i==8))
    then
        param="-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0 -gt6min 0 -gt7min 0 -gt8min 0"
    elif (($i>=9))
    then
		param="-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0 -gt6min 0 -gt7min 0 -gt8min 0 -gt9min 0"
    fi

    mkdir $experimentdir/$i
    ./ngram-count -order $i -text $datadir/training.data -write $experimentdir/$i/training.ngrams
    ./ngram-count -order $i -read $experimentdir/$i/training.ngrams -lm $experimentdir/$i/training.binary -interpolate -ukndiscount $param -write-binary-lm

	while read -r pattern
	do
        echo $pattern >$datadir/test.tmp 
		srilm=$(./ngram -order $i -lm $experimentdir/$i/training.binary -ppl $datadir/test.tmp)
		regex=".* ppl= (.+) ppl1= .*"
		if [[ $srilm =~  $regex ]]
		then
		   perplexity="${BASH_REMATCH[1]}"
		   echo $pattern"@"$i"@"$perplexity >>$experimentdir/"output_srilm"
		fi
		rm $datadir/test.tmp
	done < "$datadir/test.data"  

    rm -rf $experimentdir/$i/training.*
done
