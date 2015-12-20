rm -rf ../collections/unittest
./create-collection.x -c ../collections/unittest -i ../UnitTestData/data/training.data
#./build-index.x -c ../collections/unittest -m
./unit-test.x
