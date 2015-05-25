rm -f ../collections/unittest
./create-collection.x -c ../collections/unittest -i ../UnitTestData/data/training.data
touch ../collections/unittest/text.VOCAB
./build-index.x -c ../collections/unittest
./unit-test.x
