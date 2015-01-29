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


