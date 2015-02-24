#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"

int vocabsize = 0;
int unigramdenominator=0;
int STARTTAG=3;
int ENDTAG=4;
int unksymbol=2;

vector<int> n1;//n1[1]=#unigrams with count=1, n1[2]=#bigrams with count=1, n1[3]=#trigrams with count=1, ... index=0 is always empty
vector<int> n2;//n2[1]=#unigrams with count=2, n2[2]=#bigrams with count=2, n2[3]=#trigrams with count=2, ... index=0 is always empty
vector<int> n3;//n3[1]=#unigrams with count>=3, n3[2]=#bigrams with count>=3, n3[3]=#trigrams with count>=3, ... index=0 is always empty
vector<int> n4;//n4[1]=#unigrams with count>=4, n4[2]=#bigrams with count>=4, n4[3]=#trigrams with count>=4, ... index=0 is always empty

vector<double> Y;//Y[1] is Y for unigram, Y[2] is Y for bigram,...index=0 is always empty
vector<double> D1;//D[1]=D1 for unigram, D[2]=D1 for bigram,...index=0 is always empty
vector<double> D2;//D2[1]=D2 for unigram, D2[2]=D2 for bigram,...index=0 is always empty
vector<double> D3;//D3[1]=D+3 for unigram, D3[2]=D3+ for bigram,...index=0 is always empty

int freq;

typedef struct cmdargs {
    std::string pattern_file;
    std::string collection_dir;
    std::int ngramsize;
    std::bool ismkn;
} cmdargs_t;

void
print_usage(const char* program)
{
    fprintf(stdout,"%s -c <collection dir> -p <pattern file> -m <boolean> -n <ngramsize>\n",program);
    fprintf(stdout,"where\n");
    fprintf(stdout,"  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout,"  -p <pattern file>  : the pattern file.\n");
    fprintf(stdout,"  -m <ismkn>  : the flag for Modified-KN (true), or KN (false).\n");
    fprintf(stdout,"  -n <ngramsize>  : the ngramsize integer.\n");

};

cmdargs_t
parse_args(int argc,const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    while ((op=getopt(argc,(char* const*)argv,"p:c:n:m:")) != -1) {
        switch (op) {
            case 'p':
                args.pattern_file = optarg;
                break;
            case 'c':
                args.collection_dir = optarg;
                break;
            case 'm':
		if(optarg=="true")
			args.ismkn = true;
		else
			args.ismkn=false;
                break;
            case 'n':
                args.ngramsize = stoi((string)optarg);
                break;
        }
    }
    if (args.collection_dir==""||args.pattern_file=="") {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

int freq;
/*********************************************************/
//////////////////////////////////////////////////////////
// For the extracted edge(s), returns the actual size of the
// label extracted by excluding the 1,0 added by sdsl data
// structure
///////////////////////////////////////////////////////// 
int ncomputer(int n,cst_sct3<csa_sada_int<>>::string_type pat,int size)
{	
        uint64_t lb=0, rb=cst.size()-1;
	backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
	freq = rb-lb+1;
	if(freq == 1 && lb!=rb)
	{
		freq =0;
	}	
	if(size!=0)
	{
		if(pat.size()==2&&freq>=1)
		{
			unigramdenominator++;
		}

		if(freq==1)
		{
			n1[size]+=1;
		}else if(freq==2)
		{
			n2[size]+=1;
		}else if(freq==3)
		{
			n3[size]+=1;
		}else if(freq==4)
		{
			n4[size]+=1;
		}
	}
	if(size==0)	
	{
		int ind=0;
		pat.resize(1);
		while(ind<cst.degree(cst.root()))
		{
                     auto w = cst.select_child(cst.root(),ind+1);
		     int symbol = cst.edge(w,1);
               	     if(symbol!=1&&symbol!=0)
                     {
			pat[0] = symbol;
			ncomputer(n,pat,size+1);
	             }
		     ++ind;
                 }
	  }else
	  {
		if(size+1<=args.ngramsize)
		{
			lb=0;
			rb=cst.size()-1;
       	        	backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
			freq = rb - lb + 1 ;
			if(freq==1&& lb!=rb)
			{
				freq = 0;
			}				
			if(freq>0)
			{
				auto node = cst.node(lb,rb);
			 	int depth = cst.depth(node);
				if(pat.size()==depth)
				{
					int ind=0;				
					while(ind<cst.degree(node))
			                {
			        	     auto w = cst.select_child(node,ind+1);
		        	             int symbol = cst.edge(w,depth+1);
			        	     if(symbol!=1&&symbol!=0)
				             {
						pat.push_back(symbol);
		                		ncomputer(n,pat,size+1);
						pat.pop_back();
		        		     }
				             ++ind;
				        }
			       	 }else{
					int symbol = cst.edge(node,pat.size()+1);
					if(symbol!=1&&symbol!=0)
					{				
						pat.push_back(symbol);
		        	               	ncomputer(n,pat,size+1);
						pat.pop_back();
					}
				}
			}
		}
	}
}

int calculate_denominator_rev(cst_sct3<csa_sada_int<>>::string_type pat,uint64_t lb, uint64_t rb)
{
	int denominator=0;
        auto v = cstrev.node(lb,rb);
        int j=0;
	int size = pat.size();
	freq = rb - lb + 1;
	if(freq==1 && lb!=rb)
	{
		freq =0;
	}
	if(freq>0)
	{
                if(size ==cstrev.depth(v)){
                        int deg = cstrev.degree(v);
                        int ind = 0;
                        while(ind<deg)
                        {
                                auto w = cstrev.select_child(v,ind+1);
                                int symbol = cstrev.edge(w,size+1);
				if(symbol!=0&&symbol!=1)//TODO check
				{
					denominator++;
				}
                                ind++;
                        }
                }else{
			int symbol = cstrev.edge(v,size+1);
			if(symbol!=0&&symbol!=1)//TODO check
			{
				denominator++;
			}
                }
	}else{
	}
        return denominator;
}

int calculate_denominator(cst_sct3<csa_sada_int<>>::string_type pat,uint64_t lb, uint64_t rb)
{
	int denominator=0;
	auto v = cst.node(lb,rb);
	freq = rb - lb + 1;
	if(freq==1 && lb!=rb)
	{
		freq=0;
	}
	int j=0;
	int size = pat.size();
	if(freq>0){
		if(size ==cst.depth(v))
		{
			int deg = cst.degree(v);
	  		int ind = 0;
			while(ind<deg)
			{
				auto w = cst.select_child(v,ind+1);
				int symbol = cst.edge(w,size+1);
				if(symbol!=0&&symbol!=1)
				{
				        pat.push_back(symbol);
					cst_sct3<csa_sada_int<>>::string_type patrev = pat;
					reverse(patrev.begin(),patrev.end());
					uint64_t lbrev=0, rbrev=cstrev.size()-1;
				        backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
					auto vrev = cstrev.node(lbrev,rbrev);					
					denominator +=calculate_denominator_rev(patrev,lbrev,rbrev);
				}
				pat.pop_back();
				ind++;
			}
		}else{
			cst_sct3<csa_sada_int<>>::string_type patrev = pat;
			reverse(patrev.begin(),patrev.end());
	                uint64_t lbrev=0, rbrev=cstrev.size()-1;
	                backward_search(cstrev.csa, lbrev, rbrev, pat.begin(), pat.end(), lbrev, rbrev);
	                denominator +=calculate_denominator_rev(pat,lbrev,rbrev);
		}
	}else{}
	return denominator;
}

double pkn(cst_sct3<csa_sada_int<>>::string_type pat,bool mkn)
{
        int size = pat.size();
	
	uint64_t leftbound=0, rightbound=cst.size()-1;
	double probability=0;
	
	if( (size==args.ngramsize && args.ngramsize!=1) || (pat[0]==STARTTAG)){//for the highest order ngram, or the ngram that starts with <s>
		int c=0;
		uint64_t lb=0, rb=cst.size()-1;
        	backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
	        c = rb-lb+1;
                if(c==1 && lb!=rb)
		{
			c=0;
		}
		double D=0;
		/////// Different Discounts for mKN and KN////////////////
		if(mkn)
		{
			if(c==1){
				if(n1[args.ngramsize]!=0)	
					D = D1[args.ngramsize];
			}else if(c==2){
				if(n2[args.ngramsize]!=0)
					D = D2[args.ngramsize];
			}else if(c>=3){
				if(n3[args.ngramsize]!=0)
					D = D3[args.ngramsize];
			}
		}
	        else{
			D=Y[args.ngramsize]; 
		}
		
		/////// The same computation for both KN and mKN /////
	        double numerator=0;
        	if(c-D>0)
	        {
                	numerator = c-D;
        	}
		/////// The same computation for both KN and mKN /////
        	double denominator=0;
	        int N = 0;

		/////// backoff pattern /////////////////////////////
		cst_sct3<csa_sada_int<>>::string_type pat2 = pat;
		pat2.erase(pat2.begin());
		/////// The same denominator for both KN and mKN /////
		pat.pop_back();
		lb=0;rb=cst.size()-1;
		backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
		freq = rb - lb +1;
		if(freq==1 && lb!=rb)
		{
			freq=0;
		}
		denominator = freq;
		if(denominator==0)
	        {
			cout<<pat<<endl;
        	        cout<<"---- Undefined fractional number XXXZ - Backing-off ---"<<endl;
			double output = pkn(pat2,mkn); //TODO check this
			return output;
		}
                auto v = cst.node(lb,rb);
		if(freq>0)
		{
			if(pat.size()==cst.depth(v)){
				int ind=0;
				N=0;
				while(ind<cst.degree(v))
			      	{
		       	             auto w = cst.select_child(v,ind+1);
	      	        	     int symbol = cst.edge(w,pat.size()+1);
		       	             if(symbol!=1&&symbol!=0)
	     	                     {
					N++;
	      		             }
			             ++ind;
		                }
			}else{
				 int symbol = cst.edge(v,pat.size()+1);
				 if(symbol!=1&&symbol!=0)
	     	                 {
				     N=1;
	      		         }
			}
		}else{
			N=0;
		}
		///// The gamma for mKN //////////////////////////////
		if(mkn)
		{
			double gamma=0;
			freq = rb - lb + 1;
			if(freq==1&&rb!=lb)
			{
				freq = 0;
			}
        	        auto vtemp = cst.node(lb,rb);
			int N1=0,N2=0,N3=0;
			if(freq>0)
			{
				if(pat.size()==cst.depth(vtemp))
				{		
					int ind=0;
					while(ind<cst.degree(vtemp))
					{
						auto w = cst.select_child(vtemp,ind+1);
		                                int symbol = cst.edge(w,pat.size()+1);
						if(symbol!=1&&symbol!=0)
						{
							pat.push_back(symbol);
							leftbound=0, rightbound=cst.size()-1;
							backward_search(cst.csa, leftbound, rightbound, pat.begin(), pat.end(), lb, rb);
							freq = rightbound - leftbound + 1;
							if(freq==1&&rightbound!=leftbound)
							{
								freq = 0;
							}
							if(freq==1)
			                	                N1+=1;
			        	                else if(freq==2)
				                                N2+=1;
				                        else if(freq>=3)
		                              			N3+=1;

							pat.pop_back();
						}					
						++ind;
					}	
				}else{
		 			 int symbol = cst.edge(vtemp,pat.size()+1);
					 if(symbol!=1&&symbol!=0)
		     	                 {
						pat.push_back(symbol);

						leftbound=0, rightbound=cst.size()-1;
						backward_search(cst.csa, leftbound, rightbound, pat.begin(), pat.end(), lb, rb);
						freq = rightbound - leftbound + 1;
						if(freq==1&&rightbound!=leftbound)
						{
							freq = 0;
						}
						if(freq==1)
							N1+=1;
						else if(freq==2)
							N2+=1;
						else if(freq>=3)
							N3+=1;

						pat.pop_back();
		      		         }
				}
			}else{

			}
                        gamma = (D1[args.ngramsize] * N1) + (D2[args.ngramsize] * N2) + (D3[args.ngramsize] * N3);
			double output = (numerator/denominator) + (gamma/denominator)*pkn(pat2,mkn);
                        return output; 
		}
		else{
        	        double output= (numerator/denominator) + (D*N/denominator)*pkn(pat2,mkn);
			return output;
		}
	} else if(size<args.ngramsize&&size!=1){//for lower order ngrams

		int c=0;         
		cst_sct3<csa_sada_int<>>::string_type patrev = pat;
		reverse(patrev.begin(), patrev.end());
                uint64_t lbrev=0, rbrev=cstrev.size()-1;
                backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
		freq = rbrev - lbrev + 1;
		if(freq == 1 && lbrev!=rbrev)
		{
			freq = 0;
		}
                auto vrev = cstrev.node(lbrev,rbrev);
		// c = N1+(.abc) which is equivalent to c = N1+(cba.)	
		if(freq>0)
		{
		        if(patrev.size()==cstrev.depth(vrev)){
				int ind=0;
				c=0;
				while(ind<cstrev.degree(vrev))
			      	{
		       	             auto w = cstrev.select_child(vrev,ind+1);
	      	        	     int symbol = cstrev.edge(w,patrev.size()+1);
		       	             if(symbol!=1&&symbol!=0)
	     	                     {
					c++;
	      		             }
			             ++ind;
		                }
		        }else{
				int symbol = cstrev.edge(vrev,patrev.size()+1);
				if(symbol!=1&&symbol!=0)
	     	                {
					c=1;
				}
		        }
		}else{
		}
		double D=0;
		/////// Different Discounts for mKN and KN////////////////
                if(mkn)
                {
                        if(freq==1){
				if(n1[size]!=0)
					D = D1[size];
                        }else if(freq==2){
				if(n2[size]!=0)
					D = D2[size];
                        }else if(freq>=3){
				if(n3[size]!=0)
					D = D3[size];
			}
                }
                else{
                        D=Y[args.ngramsize];                                            
                }

	        double numerator=0;
        	if(c-D>0)
	        {
                	numerator = c-D;
        	}
		////////// backoff pattern ///////////////////////
		cst_sct3<csa_sada_int<>>::string_type pat3=pat;
 		pat3.erase(pat3.begin());
	
                pat.pop_back();
		uint64_t lb=0, rb=cst.size()-1;
                backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
                auto v = cst.node(lb,rb);
		freq = rb - lb + 1;
		if(freq==1&&lb!=rb)
		{
			freq = 0 ;
		}
		double denominator = calculate_denominator(pat,lb,rb);
                if(denominator==0)
                {
			double output = pkn(pat3,mkn);
			return output;//TODO check
                }		

	        int N=0;
		if(!mkn)
		{
			if(freq>0)
			{
				if(pat.size()==cst.depth(v)){
					int ind=0;
					while(ind<cst.degree(v))
				      	{
			       	             auto w = cst.select_child(v,ind+1);
		      	        	     int symbol = cst.edge(w,pat.size()+1);
			       	             if(symbol!=1&&symbol!=0)
		     	                     {
						N++;
		      		             }
				             ++ind;
			                }
			        }else{
					int symbol = cst.edge(v,pat.size()+1);
					if(symbol!=1&&symbol!=0)
		     	                {
						N=1;
					}
			        }
			}else{
			}
		}

		if(mkn)
		{
			 double gamma = 0;
                         auto vtemp = cst.node(lb,rb);
			 freq = rb - lb + 1;
			 if(freq==1&&lb!=rb)
				freq = 0;
                         int N1=0,N2=0,N3=0;
			 if(freq>0)
			 {
		                 if(pat.size()==cst.depth(vtemp))
		                 {
		                        int ind=0;
					
		                        while(ind<cst.degree(vtemp))
		                        {
		                                auto w = cst.select_child(vtemp,ind+1);
						int symbol = cst.edge(w,pat.size()+1);
			       			if(symbol!=1&&symbol!=0)
						{		                              
							pat.push_back(symbol);
							backward_search(cst.csa, leftbound, rightbound, pat.begin(), pat.end(), lb, rb);
							freq = rightbound - leftbound + 1;
							if(freq==1&&leftbound!=rightbound)
								freq = 0;
			                                if(freq==1)
			                                        N1+=1;
			                                else if(freq==2)
			                                        N2+=1;
			                                else if(freq>=3)
			                                        N3+=1;
					
			                                pat.pop_back();
						}
		                                ++ind;
		                        }
		                 }else{
					int symbol = cst.edge(vtemp,pat.size()+1);
					if(symbol!=1&&symbol!=0)
		     	                {
			                        pat.push_back(symbol);
						backward_search(cst.csa, leftbound, rightbound, pat.begin(), pat.end(), rb, lb);
						freq = rightbound - leftbound + 1;
						if(freq==1&&leftbound!=rightbound)
							freq = 0;
			                        if(freq==1)
			                                N1+=1;
			                        else if(freq==2)
			                                N2+=1;
			                        else if(freq>=3)
			                                N3+=1;

			                        pat.pop_back();
			                }
		                 }
			 }else{
			 }
                         gamma = (D1[size] * N1) + (D2[size] * N2) + (D3[size] * N3);
			 double output = numerator/denominator + (gamma/denominator)*pkn(pat3,mkn);
			 return output;
		}
                else{
			double output = (numerator/denominator) + (D*N/denominator)*pkn(pat3,mkn);
			return output;
		}
	}else if(size==1 || args.ngramsize == 1)//for unigram
	{
		int c=0;     
		cst_sct3<csa_sada_int<>>::string_type patrev = pat;
   		reverse(patrev.begin(), patrev.end());

                uint64_t lbrev=0, rbrev=cstrev.size()-1;
                backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
		freq = rbrev - lbrev + 1;		
		if(freq==1&&lbrev!=rbrev)
			freq =0;
                auto vrev = cstrev.node(lbrev,rbrev);
		if(freq>0)
		{
		        if(patrev.size()==cstrev.depth(vrev)){
				
				int ind=0;
				c=0;
		                while(ind<cstrev.degree(vrev))
		                {
		                	auto w = cstrev.select_child(vrev,ind+1);
					int symbol = cstrev.edge(w,patrev.size()+1);
			       		if(symbol!=1&&symbol!=0)
					{
			        	        ++c;
					}  
					ind++;        
		                }
		        }else{
				int symbol = cstrev.edge(vrev,patrev.size()+1);
				if(symbol!=1&&symbol!=0)
				{
			                c=1;
				}          
		        }
		}else{
		}
        	double denominator=unigramdenominator;

                if(denominator==0)
                {
                        cout<<"---- Undefined fractional number XXXXY-backing-off---"<<endl;
			double output = (1/(double)vocabsize);//TODO check this.
			return output;
                }
		double output = c/denominator;
		return output;
	}
	return probability;
}




template<class t_idx>
double run_query_knm(const t_idx& idx,const std::vector<uint64_t>& word_vec)
{
    double final_score=1;
    std::deque<uint64_t> pattern;
    for(const auto& word : word_vec) {
        pattern.push_front(word);
        double score = stupidbackoff(idx.m_cst_rev.csa, pattern);
        final_score*=score;
    }
    return final_score;
}



template<class t_idx>
void run_queries(t_idx& idx,const std::string& col_dir,const std::vector<std::vector<uint64_t>> patterns)
{
    using clock = std::chrono::high_resolution_clock;
    auto index_file = col_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if(utils::file_exists(index_file)) {
        std::cout << "loading index from file '" << index_file << "'" << std::endl;
        sdsl::load_from_file(idx,index_file);
        ////TODO: XXXXXX EHSAN : the reverse that you automatically generate needs to be loaded


	vocabsize = cst.degree(cst.root())-3;// -3 is for deducting the count for 0, and 1, and 3
	n1.resize(ngramsize+1);
	n2.resize(ngramsize+1);
	n3.resize(ngramsize+1);
	n4.resize(ngramsize+1);

	//precompute n1,n2,n3,n4
	cst_sct3<csa_sada_int<>>::string_type pat(1);
	ncomputer(ngramsize, pat, 0);

	Y.resize(ngramsize+1);
	if(ismkn)
	{
		D1.resize(ngramsize+1);
		D2.resize(ngramsize+1);
		D3.resize(ngramsize+1);
	}
	//compute discounts Y, D1, D2, D+3
	for(int size=1;size<=ngramsize;size++)
	{
                Y[size]=(double)n1[size]/(n1[size]+2*n2[size]);
		if(ismkn)
		{
                       	if(n1[size]!=0)
                       		D1[size] = 1 - 2 * Y[size] * (double)n2[size]/n1[size];
                	if(n2[size]!=0)
       	                	D2[size] = 2 - 3 * Y[size] * (double)n3[size]/n2[size];
               		if(n3[size]!=0)
               	        	D3[size] = 3 - 4 * Y[size] * (double)n4[size]/n3[size];
		}
	}

	cout<<"------------------------------------------------"<<endl;
        cout<<n1<<endl;
        cout<<n2<<endl;
        cout<<n3<<endl;
	cout<<n4<<endl;
	cout<<"------------------------------------------------"<<endl;
        for(int size=1;size<=ngramsize;size++)
        {
      	      cout<<"n= "<<size<<" Y= "<<Y[size]<<endl;
	      if(ismkn)
	      		cout<<"\t"<<size<<" D1= "<<D1[size]<<" D2= "<<D2[size]<<" D3= "<<D3[size]<<endl;
        }

	cout<<"N+1(..)= "<<unigramdenominator<<endl;
	cout<<"-------------------------------"<<endl;



/*
	ifstream file(path);
	string line;
	double perplexity=0;
	int M = 0; // number of words in the test data

	cst_sct3<csa_sada_int<>>::string_type temppattern(1);
	clock_t start_prec = clock();
	
	while (getline(file,line))
	{
		double sentenceprobability=0;
		vector<int> word_vec;
		istringstream iss(line);
		string word;
		//loads the a test sentence into a vector of words
		word_vec.push_back(STARTTAG);
	       	while (std::getline(iss, word, ' '))
	       	{
	      		int num = stoi(word);
	      		word_vec.push_back(num);
			M++;
	       	}
        	word_vec.push_back(ENDTAG);
		M++;//for </s> tag
        	//generates all the required patterns from the word vector
       		deque<int> pattern_deq;
        	for (auto it = word_vec.begin(); it != word_vec.end(); ++it)
        	{
	       		pattern_deq.push_back(*it);
			if(*it==STARTTAG)
				continue;
	       		while (pattern_deq.size() > ngramsize)
	       		{
			        pattern_deq.pop_front();
			}
			cst_sct3<csa_sada_int<>>::string_type pattern(pattern_deq.begin(), pattern_deq.end());
			double prob = pkn(pattern,ismkn);
			sentenceprobability = sentenceprobability+log10(prob);			
		}
		perplexity+=sentenceprobability;
	}

	clock_t end_prec = clock();
        double elapsed_secs_prec = double(end_prec - start_prec) / CLOCKS_PER_SEC;
        cout << "Query time - Time in seconds: "<<elapsed_secs_prec<<endl;

	perplexity = (1/(double)M) * perplexity;
	perplexity = pow(10,(-perplexity));
	cout<<"*************************************************"<<endl;
	if(ismkn)
		cout<<"Modified Kneser-Ney"<<endl;
	else
		cout<<"Kneser-Ney"<<endl;
	cout<<"Perplexity is: "<<perplexity<<endl;
	cout<<"*************************************************"<<endl;
*/


        std::chrono::nanoseconds total_time(0);
        for(const auto& pattern : patterns) {
            // run the query
            auto start = clock::now();
            double score = run_query_knm(idx,pattern);
            auto stop = clock::now();
            // output score
            std::copy(pattern.begin(),pattern.end(), std::ostream_iterator<uint64_t>(std::cout, " "));
            std::cout << " -> " << score;
            total_time += (stop-start);
        }
        std::cout << "time in milliseconds = " 
            << std::chrono::duration_cast<std::chrono::microseconds>(total_time).count()/1000.0f 
            << " ms" << endl;
    } else {
        std::cerr << "index does not exist. build it first" << std::endl;
    }
}



int main(int argc,const char* argv[])
{
	using clock = std::chrono::high_resolution_clock;

    /* parse command line */
    cmdargs_t args = parse_args(argc,argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* load index */
    using csa_type = sdsl::csa_sada_int<>;
    using cst_type = sdsl::cst_sct3<csa_type>;
    index_succinct<cst_type> idx;
    auto index_file = args.collection_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    std::cout << "loading index from file '" << index_file << "'" << std::endl;
    sdsl::load_from_file(idx,index_file);

    /* parse pattern file */
    std::vector< std::vector<uint64_t> > patterns;
    if(utils::file_exists(args.pattern_file)) {
        std::ifstream ifile(args.pattern_file);
        std::cout << "reading input file '" << args.pattern_file << "'" << std::endl;
        std::string line;
        while (std::getline(ifile,line)) {
            std::vector<uint64_t> tokens;
            std::istringstream iss(line);
            std::string word;
            while (std::getline(iss, word, ' ')) {
                uint64_t num = std::stoull(word);
                tokens.push_back(num);
            }
	    patterns.push_back(tokens);
        }
    } else {
        std::cerr << "cannot read pattern file '" << args.pattern_file << "'" << std::endl;
        return EXIT_FAILURE;
    }
    
    {

	//TODO XXXXX EHSAN : how to load the reverse data structure? XXXXXXXXX
	using wt_bv_rank_1_type = rank_support_v5<1,1>;
    	using wavelet_tree_type = wt_huff_int<wt_bitvector_type,wt_bv_rank_1_type,select_support_scan<1,1>, select_support_scan<0,1>>;
    	const uint32_t sa_sample_rate = 32;
    	const uint32_t isa_sample_rate = 64;
        using csa_type = csa_wt_int<wavelet_tree_type,sa_sample_rate,isa_sample_rate> csarev;
        index_succinct<csa_type> idx;
        run_queries(idx,args.collection_dir,patterns);
    }


    return 0;
}
