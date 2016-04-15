Welcome to CSTLM
================

This is a compressed suffix tree based infinite context size language model capable of indexing terabyte sized text collections.

# References

This code is the basis of the following papers:

- Ehsan Shareghi, Matthias Petri, Gholamreza Haffari, Trevor Cohn: Compact, Efficient and Unlimited Capacity: Language Modeling with Compressed Suffix Trees. EMNLP 2015: 2409-2418

- TBA

# Compile instructions

1. Check out the reprository: `https://github.com/mpetri/cstlm.git`
2. `git submodule update --init`
3. `cd build`
4. `cmake ..`
5. `make -j`

# Run unit tests to ensure correctness

```
cd build
rm -rf ../collections/unittest/
./create-collection.x -i ../UnitTestData/data/training.data -c ../collections/unittest
./create-collection.x -i ../UnitTestData/data/training.data -c ../collections/unittest -1
./unit-test.x
```

# Usage instructions (Word based language model)

Create collection:

```
./create-collection.x -i toyfile.txt -c ../collections/toy
```

Build index (including quantities for modified KN)

```
./build-index.x -c ../collections/toy/ -m
```

# Usage instructions (Character based language model)

Create collection:

```
./create-collection.x -i toyfile.txt -c ../collections/toy -1
```

Build index (including quantities for modified KN)

```
./build-index.x -c ../collections/toy/ -m
```

# Moses integration  (Word based language model)

Compile moses using

```
./compile.sh --with-cstlm=<path to repo>
```

Create the collection and build the index for the monolingual corpus

```
./create-collection.x -i mono.txt -c ../collections/mono
./build-index.x -c ../collections/mono/ -m

```

Modify moses.ini and replace the KENLM line with

```
CSTLM-WORD factor=0 order=10 path=<path to collection>/collections/mono/
```

# Moses integration  (Character based language model)

Compile moses using

```
./compile.sh --with-cstlm=<path to repo>
```

Create the collection and build the index for the monolingual corpus

```
./create-collection.x -i mono.txt -c ../collections/mono -1
./build-index.x -c ../collections/mono/ -m

```

Modify moses.ini and replace the KENLM line with

```
CSTLM-CHAR factor=0 order=50 path=<path to collection>/collections/mono/
```

