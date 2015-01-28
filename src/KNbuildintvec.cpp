#include <sdsl/suffix_trees.hpp>
#include <iostream>
#include <fstream>
#include <stdint.h>

using namespace sdsl;
using namespace std;

int linecounter(string path)
{
	int number_of_lines=0;
	ifstream infile(path);
	string line;
	while (getline(infile, line))
	        ++number_of_lines;
	return number_of_lines;
}

int wordcounter(string path)
{
        ifstream infile(path);
        istream_iterator<string> in{ infile }, end;
        int count= distance(in, end);
	return count;
}

int originalorder(string path,int count)
{
	ifstream file(path);
	string line;
	sdsl::int_vector<> v(count);
	int i=0;
	while (getline(file,line))
	{
	//	vector<int> vec(1);
		istringstream iss(line);
		string word;
	//	int size=0;
		while (std::getline(iss, word, ' '))
		{
			int num = stoi(word);
	//		vec.resize(size+1);
			v[i]=num;
	//		++size;
			++i;
		}
		v[i]=1;
		++i;
	}
//	cout<<v<<endl;
	util::bit_compress(v);
	store_to_file(v,"file.sdsl");
	return 0;
}

int reversedorder(string path,int count)
{
	ifstream file(path);
	string line;
	sdsl::int_vector<> v(count);
	int i=0;
	while (getline(file,line))
	{
		vector<int> vec(1);
		istringstream iss(line);
		string word;
		int size=1;
		while (std::getline(iss, word, ' '))
		{
			int num = stoi(word);
			vec.resize(size);
			vec[size-1]=num;
		//	v[i]=num;
		//	++i;
			++size;
		}
		while(size>1)
		{
			v[i]=vec[size-2];
			--size;
			++i;
		}
		v[i]=1;
		++i;

	}
//	cout<<v<<endl;
	util::bit_compress(v);

	store_to_file(v,"filereversed.sdsl");
}

int main(int argc,char *argv[])
{
	if(argc < 2) {
                printf("You must provide data path\n");
                exit(0);
        }

//      string path = "../data/d.txt";
        string path = argv[1];

        int wordcount = wordcounter(path); //number of words in the file
        int linecount=linecounter(path); // number of lines in the file (=number )

        int count = wordcount + linecount;// words+ number of 1s (0 will be added automatically to the end of the vector)
	
	originalorder(path,count);
	reversedorder(path,count);
 	return 0;
}
