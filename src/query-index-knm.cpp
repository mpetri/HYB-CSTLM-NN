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

int ngramsize;
int vocabsize = 0;
int unigramdenominator=0;
int STARTTAG=3;
int ENDTAG=4;
int unksymbol=2;
bool ismkn;

vector<int> n1;//n1[1]=#unigrams with count=1, n1[2]=#bigrams with count=1, n1[3]=#trigrams with count=1, ... index=0 is always empty
vector<int> n2;//n2[1]=#unigrams with count=2, n2[2]=#bigrams with count=2, n2[3]=#trigrams with count=2, ... index=0 is always empty
vector<int> n3;//n3[1]=#unigrams with count=3, n3[2]=#bigrams with count=3, n3[3]=#trigrams with count=3, ... index=0 is always empty
vector<int> n4;//n4[1]=#unigrams with count>=4, n4[2]=#bigrams with count>=4, n4[3]=#trigrams with count>=4, ... index=0 is always empty
int N3plus;

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

int ncomputer(const t_idx& idx,cst_sct3<csa_sada_int<>>::string_type pat,int size)
{	
        uint64_t lb=0, rb=idx.m_cst.size()-1;
	backward_search(idx.m_cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
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
		}else if(freq>=3)
		{
			if(freq==3)
			{
				n3[size]+=1;
			}else if(freq==4)
			{
				n4[size]+=1;
			}
			if(size==1) 
				N3plus++;
		}
	}
	if(size==0)	
	{
		int ind=0;
		pat.resize(1);
		while(ind<idx.m_cst.degree(idx.m_cst.root()))
		{
                     auto w = idx.m_cst.select_child(idx.m_cst.root(),ind+1);
		     int symbol = idx.m_cst.edge(w,1);
               	     if(symbol!=1&&symbol!=0)
                     {
			pat[0] = symbol;
			ncomputer(idx,pat,size+1);
	             }
		     ++ind;
               }
	  }else
	  {
		if(size+1<=ngramsize)
		{
			if(freq>0)
			{
				auto node = idx.m_cst.node(lb,rb);
			 	int depth = idx.m_cst.depth(node);
				if(pat.size()==depth)
				{
					int ind=0;
					
					while(ind<idx.m_cst.degree(node))
		                	{
		        	             auto w = idx.m_cst.select_child(node,ind+1);
	        	        	     int symbol = idx.m_cst.edge(w,depth+1);
		        	             if(symbol!=1&&symbol!=0)
			                     {
						pat.push_back(symbol);
	                		        ncomputer(idx, pat,size+1);
					        pat.pop_back();
	        		             }
			                     ++ind;
			                }
		       	        }else{
					int symbol = idx.m_cst.edge(node,pat.size()+1);
					if(symbol!=1&&symbol!=0)
					{				
						pat.push_back(symbol);
	        	                	ncomputer(idx, pat,size+1);
						pat.pop_back();
					}
				}
			}else{
			}
		}
	}
}

int calculate_denominator_rev(const t_idx& idx,cst_sct3<csa_sada_int<>>::string_type pat,uint64_t lb, uint64_t rb)
{
	int denominator=0;
        auto v = idx.m_cst_rev.node(lb,rb);
	int size = pat.size();
	freq = rb - lb + 1;
	if(freq==1 && lb!=rb)
	{
		freq =0;
	}
	if(freq>0)
	{
                if(size ==idx.m_cst_rev.depth(v)){
                        int deg = idx.m_cst_rev.degree(v);
                        int ind = 0;
                        while(ind<deg)
                        {
                                auto w = idx.m_cst_rev.select_child(v,ind+1);
                                int symbol = idx.m_cst_rev.edge(w,size+1);
				if(symbol!=0&&symbol!=1)//TODO check
				{
					denominator++;
				}
                                ind++;
                        }
                }else{
			int symbol = idx.m_cst_rev.edge(v,size+1);
			if(symbol!=0&&symbol!=1)//TODO check
			{
				denominator++;
			}
                }
	}else{
	}
        return denominator;
}

int calculate_denominator(const t_idx& idx,cst_sct3<csa_sada_int<>>::string_type pat,uint64_t lb, uint64_t rb)
{
	int denominator=0;
	auto v = idx.m_cst.node(lb,rb);
	freq = rb - lb + 1;
	if(freq==1 && lb!=rb)
	{
		freq=0;
	}
	int size = pat.size();
	if(freq>0){
		if(size ==idx.m_cst.depth(v))
		{
			int deg = idx.m_cst.degree(v);
	  		int ind = 0;
			while(ind<deg)
			{
				auto w = idx.m_cst.select_child(v,ind+1);
				int symbol = idx.m_cst.edge(w,size+1);
				if(symbol!=0&&symbol!=1)
				{
				        pat.push_back(symbol);
					cst_sct3<csa_sada_int<>>::string_type patrev = pat;
					reverse(patrev.begin(),patrev.end());
					uint64_t lbrev=0, rbrev=idx.m_cst_rev.size()-1;
				        backward_search(idx.m_cst_rev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);					
					denominator +=calculate_denominator_rev(patrev,lbrev,rbrev);
					pat.pop_back();
				}
				ind++;
			}
		}else{
			cst_sct3<csa_sada_int<>>::string_type patrev = pat;
			reverse(patrev.begin(),patrev.end());
	                uint64_t lbrev=0, rbrev=idx.m_cst_rev.size()-1;
	                backward_search(idx.m_cst_rev.csa, lbrev, rbrev, pat.begin(), pat.end(), lbrev, rbrev);
	                denominator +=calculate_denominator_rev(pat,lbrev,rbrev);
		}
	}else{}
	return denominator;
}

double pkn(const t_idx& idx,cst_sct3<csa_sada_int<>>::string_type pat)
{
        int size = pat.size();
	
	uint64_t leftbound=0, rightbound=idx.m_cst.size()-1;
	double probability=0;
	
	if( (size==ngramsize && ngramsize!=1) || (pat[0]==STARTTAG)){//for the highest order ngram, or the ngram that starts with <s>
		int c=0;
		uint64_t lb=0, rb=idx.m_cst.size()-1;
        	backward_search(idx.m_cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
	        c = rb-lb+1;
                if(c==1 && lb!=rb)
		{
			c=0;
		}
		double D=0;
		if(ismkn)
		{
			if(c==1){
				if(n1[ngramsize]!=0)	
					D = D1[ngramsize];
			}else if(c==2){
				if(n2[ngramsize]!=0)
					D = D2[ngramsize];
			}else if(c>=3){
				if(n3[ngramsize]!=0)
					D = D3[ngramsize];
			}
		}
	        else{
			D=Y[ngramsize]; 
		}
		
	        double numerator=0;
        	if(c-D>0)
	        {
                	numerator = c-D;
        	}

        	double denominator=0;
	        int N = 0;


		cst_sct3<csa_sada_int<>>::string_type pat2 = pat;
		pat2.erase(pat2.begin());

		pat.pop_back();
		lb=0;rb=idx.m_cst.size()-1;
		backward_search(idx.m_cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
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
			double output = pkn(idx,pat2); //TODO check this
			return output;
		}
                auto v = idx.m_cst.node(lb,rb);
                int N1=0,N2=0,N3=0;
		int pat_size = pat.size();
		if(freq>0)
		{
			if(pat_size==idx.m_cst.depth(v)){
				int ind=0;
				N=0;
				while(ind<idx.m_cst.degree(v))
			      	{
		       	             auto w = idx.m_cst.select_child(v,ind+1);
	      	        	     int symbol = idx.m_cst.edge(w,pat_size+1);
		       	             if(symbol!=1&&symbol!=0)
	     	                     {
					N++;
					if(ismkn)
					{
	                                        pat.push_back(symbol);
                                                leftbound=0, rightbound=idx.m_cst.size()-1;
                                                backward_search(idx.m_cst.csa, leftbound, rightbound, pat.begin(), pat.end(), lb, rb);
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
			             ++ind;
		                }
			}else{
				 int symbol = idx.m_cst.edge(v,pat.size()+1);
				 if(symbol!=1&&symbol!=0)
	     	                 {
				     N=1;
				     if(ismkn)
				     {
				     		pat.push_back(symbol);

                                                leftbound=0, rightbound=idx.m_cst.size()-1;
                                                backward_search(idx.m_cst.csa, leftbound, rightbound, pat.begin(), pat.end(), lb, rb);
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
			}
		}else{
			N=0;
		}
		if(ismkn)
		{
                        double gamma = (D1[ngramsize] * N1) + (D2[ngramsize] * N2) + (D3[ngramsize] * N3);
			double output = (numerator/denominator) + (gamma/denominator)*pkn(idx,pat2);
                        return output; 
		}
		else{
        	        double output= (numerator/denominator) + (D*N/denominator)*pkn(idx,pat2);
			return output;
		}
	} else if(size<ngramsize&&size!=1){//for lower order ngrams

		int c=0;         
		cst_sct3<csa_sada_int<>>::string_type patrev = pat;
		reverse(patrev.begin(), patrev.end());
                uint64_t lbrev=0, rbrev=idx.m_cst_rev.size()-1;
                backward_search(idx.m_cst_rev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
		freq = rbrev - lbrev + 1;
		if(freq == 1 && lbrev!=rbrev)
		{
			freq = 0;
		}
                auto vrev = idx.m_cst_rev.node(lbrev,rbrev);
		int patrev_size = patrev.size();

		if(freq>0)
		{
		        if(patrev.size()==idx.m_cst_rev.depth(vrev)){
				int ind=0;
				c=0;
				while(ind<idx.m_cst_rev.degree(vrev))
			      	{
		       	             auto w = idx.m_cst_rev.select_child(vrev,ind+1);
	      	        	     int symbol = idx.m_cst_rev.edge(w,patrev_size+1);
		       	             if(symbol!=1&&symbol!=0)
	     	                     {
					c++;
	      		             }
			             ++ind;
		                }
		        }else{
				int symbol = idx.m_cst_rev.edge(vrev,patrev_size+1);
				if(symbol!=1&&symbol!=0)
	     	                {
					c=1;
				}
		        }
		}else{
		}
		double D=0;

                if(ismkn)
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
                        D=Y[ngramsize];                                            
                }

	        double numerator=0;
        	if(c-D>0)
	        {
                	numerator = c-D;
        	}

		cst_sct3<csa_sada_int<>>::string_type pat3=pat;
 		pat3.erase(pat3.begin());
		
                pat.pop_back();		

		uint64_t lb=0, rb=idx.m_cst.size()-1;
                backward_search(idx.m_cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
		freq = rb - lb + 1;
		if(freq==1&&lb!=rb)
		{
			freq = 0 ;
		}
		double denominator=0;
		if(freq!=1)
		{
			denominator= calculate_denominator(pat,lb,rb);
		}else{
			denominator = 1;
		}		
                if(denominator==0)
                {
                        cout<<"---- Undefined fractional number XXXW-backing-off---"<<endl;
			double output = pkn(idx,pat3);
			return output;//TODO check
                }
                auto v = idx.m_cst.node(lb,rb);
		


		if(ismkn)
		{
			 double gamma = 0;
                         int N1=0,N2=0,N3=0;
			 if(freq>0)
			 {
		                 if(pat.size()==idx.m_cst.depth(v))
		                 {
		                        int ind=0;
			
		                        while(ind<idx.m_cst.degree(v))
		                        {
		                                auto w = idx.m_cst.select_child(v,ind+1);
						int symbol = idx.m_cst.edge(w,pat.size()+1);
			       			if(symbol!=1&&symbol!=0)
						{		                              
							pat.push_back(symbol);
							backward_search(idx.m_cst.csa, leftbound, rightbound, pat.begin(), pat.end(), lb, rb);
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
					int symbol = idx.m_cst.edge(v,pat.size()+1);
					if(symbol!=1&&symbol!=0)
		     	                {
			                        pat.push_back(symbol);
						backward_search(idx.m_cst.csa, leftbound, rightbound, pat.begin(), pat.end(), rb, lb);
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
			 double output = numerator/denominator + (gamma/denominator)*pkn(idx,pat3);
			 return output;
		}else{

			int N=0;
			int pat_size = pat.size();
			if(freq>0)
			{
				if(pat.size()==idx.m_cst.depth(v)){
				int ind=0;
					while(ind<idx.m_cst.degree(v))
				      	{
			       	             auto w = idx.m_cst.select_child(v,ind+1);
		      	        	     int symbol = idx.m_cst.edge(w,pat_size+1);
			       	             if(symbol!=1&&symbol!=0)
		    	                     {
						N++;
		      		             }
				             ++ind;
			                }
			        }else{
					int symbol = idx.m_cst.edge(v,pat_size+1);
					if(symbol!=1&&symbol!=0)
			                {
						N=1;
					}
			        }
			}else{}
			double output = (numerator/denominator) + (D*N/denominator)*pkn(idx,pat3);
			return output;
		}
	}else if(size==1 || ngramsize == 1)//for unigram
	{
		int c=0;     
                uint64_t lbrev=0, rbrev=idx.m_cst_rev.size()-1;
                backward_search(idx.m_cst_rev.csa, lbrev, rbrev, pat.begin(), pat.end(), lbrev, rbrev);
		freq = rbrev - lbrev + 1;		
		if(freq==1&&lbrev!=rbrev)
			freq =0;
                auto vrev = idx.m_cst_rev.node(lbrev,rbrev);
		int pat_size = pat.size();
		if(freq>0)
		{
		        if(pat.size()==idx.m_cst_rev.depth(vrev)){
				
				int ind=0;
				c=0;
		                while(ind<idx.m_cst_rev.degree(vrev))
		                {
		                	auto w = idx.m_cst_rev.select_child(vrev,ind+1);
					int symbol = idx.m_cst_rev.edge(w,pat_size+1);
			       		if(symbol!=1&&symbol!=0)
					{
			        	        ++c;
					}  
					ind++;        
		                }
		        }else{
				int symbol = idx.m_cst_rev.edge(vrev,pat_size+1);
				if(symbol!=1&&symbol!=0)
				{
			                c=1;
				}          
		        }
		}else{
		}
		
        	double denominator=unigramdenominator;
		
		if(!ismkn)
		{
			double output = c/denominator;
			return output;
		}else
		{

			double D=0;
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

			double numerator=0;
	                if(c-D>0)
        	        {
        	               numerator = c-D;
        	        }

			double gamma = 0;
			int N1=0,N2=0,N3=0;
			N1=n1[1];
			N2=n2[1];
			N1=N3plus;			               
                        gamma = (D1[size] * N1) + (D2[size] * N2) + (D3[size] * N3);
		        double output = numerator/denominator + (gamma/denominator)* (1/(double)vocabsize);  
                        return output;
		}
	}
	return probability;
}



template<class t_idx>
double run_query_knm(const t_idx& idx,const std::vector<uint64_t>& word_vec)
{
    double final_score=0;
    std::deque<uint64_t> pattern;
    for(const auto& word : word_vec) {
        pattern.push_back(word);//TODO check the pattern
	if(word==STARTTAG)
		continue;
 	if (pattern_deq.size() > ngramsize)
	{
		pattern_deq.pop_front();
	}
	cst_sct3<csa_sada_int<>>::string_type pattern(pattern_deq.begin(), pattern_deq.end());	
        double score = pkn(idx, pattern);
        final_score+=log10(score);
    }

	
		}

    return final_score;
}



template<class t_idx>
void run_queries(const t_idx& idx,const std::string& col_dir,const std::vector<std::vector<uint64_t>> patterns)
{
    using clock = std::chrono::high_resolution_clock;
    auto index_file = col_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if(utils::file_exists(index_file)) {
        std::cout << "loading index from file '" << index_file << "'" << std::endl;
        sdsl::load_from_file(idx,index_file);

	n1.resize(ngramsize+1);
	n2.resize(ngramsize+1);
	n3.resize(ngramsize+1);
	n4.resize(ngramsize+1);

	vocabsize = idx.m_cst.degree(idx.m_cst.root())-3;// -3 is for deducting the count for 0, and 1, and 3

	//precompute n1,n2,n3,n4
	ncomputer(idx,pat, 0);

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

	double perpelxity=0;
	double sentenceprob=0;
	int M=0;
        std::chrono::nanoseconds total_time(0);
        for(const auto& pattern : patterns) {
	    M +=pattern.size()-1;// -1 for discarding <s>
            // run the query
            auto start = clock::now();
            double sentenceprob = run_query_knm(idx,pattern);
            auto stop = clock::now();
	    perplexity + = log10(sentenceprob);
            // output score
            std::copy(pattern.begin(),pattern.end(), std::ostream_iterator<uint64_t>(std::cout, " "));
            std::cout << " -> " << sentenceprob;
            total_time += (stop-start);
        }
        std::cout << "time in milliseconds = " 
            << std::chrono::duration_cast<std::chrono::microseconds>(total_time).count()/1000.0f 
            << " ms" << endl;
	perplexity = (1/(double)M) * perplexity;
	perplexity = pow(10,(-perplexity));
	std::cout << "Perplexity = "<<perplexity<<endl;
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
	    patterns.push_back(tokens);//each pattern is a sentence <s> w1 w2 w3 ... </s>
        }
    } else {
        std::cerr << "cannot read pattern file '" << args.pattern_file << "'" << std::endl;
        return EXIT_FAILURE;
    }
    
    {
	using wt_bv_rank_1_type = rank_support_v5<1,1>;
    	using wavelet_tree_type = wt_huff_int<wt_bitvector_type,wt_bv_rank_1_type,select_support_scan<1,1>, select_support_scan<0,1>>;
    	const uint32_t sa_sample_rate = 32;
    	const uint32_t isa_sample_rate = 64;
        using csa_type = csa_wt_int<wavelet_tree_type,sa_sample_rate,isa_sample_rate>;
        using lcp_type = lcp_dac<>;
        using bp_support_type = bp_support_sada<>;
        using first_child_bv_type = bit_vector;
        using cst_type = cst_sct3< csa_type , lcp_type , bp_support_type, first_child_bv_type >;

        index_succinct<csa_type> idx;
        run_queries(idx,args.collection_dir,patterns);
    }


    return 0;
}
