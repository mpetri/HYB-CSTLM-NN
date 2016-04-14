Welcome to CSTLM

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

## Running `unit' tests ##

To run the unit-test.x binary first you need to do the following
```
rm -rf ../collections/unittest/
./create-collection.x -i ../UnitTestData/data/training.data -c ../collections/unittest
./build-index.x -c ../collections/unittest/ -m
./unit-test.x
```
