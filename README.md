Welcome to LM-SDSL

# compile instructions

1. Check out reprository
2. `git submodule init`
3. `git submodule update`
4. `cd build`
5. `cmake ..`

# usage

Create a collection:

```
./create-collection.x -i toyfile.txt -c ../collections/toy
```

Build index (including quantities for modified KN)

```
./build-index.x -c ../collections/toy/ -m
```

Query index
Run ./querk-index-knm.x to see the running time arguments. Here are some examples:

For Modified Kneser-ney (fastest index) with accurate version
```
./query-index-knm.x -c  ../collections/toy/ -p toyquery.txt -n 5 -s -m
```
For Modified Kneser-ney (fastest index) with fishy version version
```
./query-index-knm.x -c  ../collections/toy/ -p toyquery.txt -n 5 -s -m -f
```
For Kneser-Ney (fastest index)
```
./query-index-knm.x -c  ../collections/toy/ -p toyquery.txt -n 5
```
## Running `unit' tests ##

To run the unit-test.x binary first you need to do the following
```
rm -r ../collections/unittest/
./create-collection.x -i ../UnitTestData/data/training.data -c ../collections/unittest
./build-index.x -c ../collections/unittest/ -m
./unit-test.x
```
## Comparison on Europarl German KenLM v.s. Accurate v.s Fishy ##
Training raw size: 170MB

KENLM - on eu_de and 500 test sentences (includes OOVs:	80)
```
2-gram:	166.68	
default: 95MB    trie: 38MB
3-gram:	108.44
default: 402MB    trie: 168MB
4-gram:	100.75
default: 963MB    trie: 441MB
5-gram:	99.59
default: 1692MB    trie: 825MB
```
under the same training and test setting the accurate v.s. fishy:
```
2-gram:	166.68  v.s.  166.68
275MB
3-gram:	108.44  v.s.  108.09
275MB
4-gram:	100.75  v.s.  99.68
275MB
5-gram:	99.59   v.s.  97.94
275MB
```
on larger test set with 10K sentences the fishy v.s. kenlm
```
2-gram 174.80  v.s.  174.80
3-gram 112.47  v.s.  112.84
4-gram 104.78  v.s.  105.86
5-gram 103.11  v.s.  104.82
```
## Possible ways to frame the paper (ranked by feasibility based on end of December deadline) ##
(1) As LM Paper (similar to emnlp)
```
    contributions: i) speedup, ii)comparison with kenlm (state-of-the-art)
    experiments: i) exactly similar to emnlp (nothing more)
```
(2) As LM paper on Big data
```
    contributions: all of the above, and iii) showing the impact of data size v.s. model complexity on pplx
    experiments: all of the above, and ii) experiments with different training data size of English, German,
    French, Spanish: 1GB, 3GB, 7GB, 15GB, 30GB, 60GB, 125GB, (and ideally 250GB, 500GB, and 1TiB) to produce
    1 graph for each language where each having 10 plots for n=1...10gram , where x-axis is data-size, and y-axis is pplx.
```
(3) As LM paper on Big data MT experiments
```
    contributions: all the above (there won't be any new contribution here)
    experiments: either (i), or (ii) or both, and iii) MT experiments
```
## ToDos ##
