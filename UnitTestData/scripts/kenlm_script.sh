#!/bin/bash
# this script should be placed in: kenlm
# parameter @1 is the data directory
# parameter @2 is the directory to dump the output of perplexity, etc.
# parameter @3 is the percentage of the memory to be used by kenlm

args=("$@")
datadir=${args[0]} # i.e.: lm-sdsl/UnitTestData/data
experimentdir=${args[1]} # a path to dump the outputs into
                         # once finished, have a look at this file output_kenlm
                         # in this directory for the perplexity (ppl) scores

mkdir $experimentdir
for order in 2 3 4 5 6 7 8
do
    mkdir $experimentdir/$order

    bin/lmplz -o $order <$datadir/training.data > $experimentdir/$order/training.arpa
    bin/build_binary $experimentdir/$order/training.arpa $experimentdir/$order/training.binary

    while read -r pattern
    do
      echo $pattern >$datadir/test.tmp
      bin/query $experimentdir/$order/training.binary <$datadir/test.tmp >>$datadir/test.tmp
      tmp=$(cat $datadir/test.tmp)
      regex=".*including OOVs:	(.*)Perplexity.*"
      if [[ $tmp =~  $regex ]]
      then
	    perplexity="${BASH_REMATCH[1]}"	
            echo $pattern"@"$order"@"$perplexity >>$experimentdir/"kenlm_output"
      fi
      rm $datadir/test.tmp
    done < "$datadir/test.data"
    rm -rf $experimentdir/$order
done
