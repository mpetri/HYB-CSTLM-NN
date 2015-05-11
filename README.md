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

Build index

```
./build-index.x -c ../collections/toy/
```

Query index

```
./query-index-stupid.x -c ../collections/toy/ -p toyquery.txt
```

## Running integration tests ##

I've put some fairly simple test data under *UnitTestData/data/undoc_2000*. See the file there *README.undoc* to see how it was constructed. It's a character level representation of the first 1000 sentences of a corpus. This should be sufficient to test out the various counting methods without getting hit too heavily by the slow runtime of the *ncompute* method etc. 

To run this, first compile then from the build directory run:
```
./create-collection.x -i ../UnitTestData/data/undoc_2000_fr_en_sample.train -c ../collections/undoc
```
which will create the *undoc* folder and put a single file in there. Next run
``` 
./build-index.x -c ../collections/undoc
```
which builds the index and various precomputed stuff. Finally query with
```
./query-index-knm.x -c ../collections/undoc -p ../UnitTestData/data/undoc_2000_fr_en_sample.test -n 5
```
which queries using 5-grams for a test sample. There are a couple of issues here, namely it reports:
```sh
------------------------------------------------
-------------PRECOMPUTED QUANTITIES-------------
------------------------------------------------
n1 = 0 4 254 1898 6734 15697 
n2 = 0 2 133 852 2523 5218 
n3 = 0 1 105 437 1519 2745 
n4 = 0 1 72 317 974 1598 
------------------------------------------------
Y = 0 0.5 0.488462 0.526929 0.571647 0.600658 
------------------------------------------------
N1+(..) = 1394
N3+(.) = 116
------------------------------------------------
------------------------------------------------
reading input file '../UnitTestData/data/undoc_2000_fr_en_sample.test'
------------------------------------------------
------------------------------------------------
PATTERN is = 3 26 
dot_LB= 135477  dot_RB= 135654
Lowest Order numerator is: 10 denomiator is: 1394
Lowest Order probability 0.0071736
------------------------------------------------
dot_LB= 135477  dot_RB= 135654
0x7fc10860f720
0x7fc10860f720
XXXdot_LB= 135478  dot_RB= 135505
dot_LB= 135477  dot_RB= 135654
XXXdot_LB= 2001  dot_RB= 2000
```
where the leading 0s in the precomputed quantities seem fishy, and the last few lines look like some kind of  failure case.
