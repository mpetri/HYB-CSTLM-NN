#include <ctime>
#include <unistd.h>
#include <ios>
#include <fstream>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include <sdsl/suffix_trees.hpp>
#include <iostream>
#include <string>
#include <math.h>

using namespace sdsl;
using namespace std;

cst_sct3<csa_sada_int<>> cst;
cst_sct3<csa_sada_int<>> cstrev;

int ngramsize;
int vocabsize = 0;
int unigramdenominator=0;

vector<int> n1;//n1[1]=#unigrams with count=1, n1[2]=#bigrams with count=1, n1[3]=#trigrams with count=1, ... index=0 is always empty
vector<int> n2;//n2[1]=#unigrams with count=2, n2[2]=#bigrams with count=2, n2[3]=#trigrams with count=2, ... index=0 is always empty
vector<int> n3;//n3[1]=#unigrams with count>=3, n3[2]=#bigrams with count>=3, n3[3]=#trigrams with count>=3, ... index=0 is always empty
vector<int> n4;//n4[1]=#unigrams with count>=4, n4[2]=#bigrams with count>=4, n4[3]=#trigrams with count>=4, ... index=0 is always empty

vector<double> Y;//Y[1] is Y for unigram, Y[2] is Y for bigram,...index=0 is always empty
vector<double> D1;//D[1]=D1 for unigram, D[2]=D1 for bigram,...index=0 is always empty
vector<double> D2;//D2[1]=D2 for unigram, D2[2]=D2 for bigram,...index=0 is always empty
vector<double> D3;//D3[1]=D+3 for unigram, D3[2]=D3+ for bigram,...index=0 is always empty

/*********************************************************/
//////////////////////////////////////////////////////////
// For the extracted edge(s), returns the actual size of the
// label extracted by excluding the 1,0 added by sdsl data
// structure
///////////////////////////////////////////////////////// 
int ncomputer(int n,cst_sct3<csa_sada_int<>>::string_type pat,int size)
{	
        uint64_t lb=0, rb=cst.size()-1;
	if(size!=0)
	{
		if(pat.size()==2&&count(cst,pat)>=1)
		{
			unigramdenominator++;
		}

		if(count(cst,pat)==1)
		{
			n1[size]+=1;
		}else if(count(cst,pat)==2)
		{
			n2[size]+=1;
		}else if(count(cst,pat)==3)
		{
			n3[size]+=1;
		}else if(count(cst,pat)==4)
		{
			n4[size]+=1;
		}
	}
//	}else{
		if(size==0)	
		{
			lb=0;
			rb=cst.size()-1;
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
			if(size+1<=ngramsize)
			{
				lb=0;
				rb=cst.size()-1;
        	        	backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
				auto node = cst.node(lb,rb);
				if(count(cst,pat)>0)
				{
				 	int depth = cst.depth(node);
					if(pat.size()==depth)
					{
						int ind=0;
						pat.resize(pat.size()+1);
						while(ind<cst.degree(node))
			                	{
			        	             auto w = cst.select_child(node,ind+1);
		        	        	     int symbol = cst.edge(w,depth+1);
			        	             if(symbol!=1&&symbol!=0)
				                     {
							pat[pat.size()-1] = symbol;
		                		        ncomputer(n,pat,size+1);
		        		             }
				                     ++ind;
				                }
			       	        }else{
						int symbol = cst.edge(node,pat.size()+1);
						if(symbol!=1&&symbol!=0)
						{
							pat.resize(size+1);
							pat[pat.size()-1]=symbol;
		        	                	ncomputer(n,pat,size+1);
						}
					}
				}else{
				}
			}
		}
//	}
}


int calculate_denominator_rev(cst_sct3<csa_sada_int<>>::string_type pat,uint64_t lb, uint64_t rb)
{
	int denominator=1;
        auto v = cstrev.node(lb,rb);
        int j=0;
	int size = pat.size();
	if(count(cstrev,pat)>0)
	{
                if(size ==cstrev.depth(v)){
                        int deg = cstrev.degree(v);
                        int ind = 0;
                        while(ind<deg)
                        {
                                auto w = cstrev.select_child(v,ind+1);
                                int symbol = cstrev.edge(w,size+1);
				if(symbol!=0&&symbol!=1)
				{
					denominator++;
				}
                                ind++;
                        }
                }else{
                }
	}else{
	}
        return denominator;
}

int calculate_denominator(cst_sct3<csa_sada_int<>>::string_type pat,uint64_t lb, uint64_t rb)
{
	int denominator=0;
	auto v = cst.node(lb,rb);
	int j=0;
	int size = pat.size();
	if(count(cst,pat)>0){
		if(size ==cst.depth(v))
		{
			int deg = cst.degree(v);
	  		int ind = 0;
			pat.resize(size+1);
			while(ind<deg)
			{
				auto w = cst.select_child(v,ind+1);
				int symbol = cst.edge(w,size+1);
				if(symbol!=0&&symbol!=1)
				{
				        pat[size]=symbol;
					cst_sct3<csa_sada_int<>>::string_type patrev(pat.size());
				        for(int i=0;i<pat.size();i++)
				        {
				                patrev[i]=pat[pat.size()-1-i];
				        }
					uint64_t lbrev=0, rbrev=cstrev.size()-1;
				        backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
					auto vrev = cstrev.node(lbrev,rbrev);
					denominator +=calculate_denominator_rev(patrev,lbrev,rbrev);
				}
				ind++;
			}
		}else{
			cst_sct3<csa_sada_int<>>::string_type patrev(pat.size());
	                for(int i=0;i<pat.size();i++)
	                {
	                     patrev[i]=pat[pat.size()-1-i];
	                }
	                uint64_t lbrev=0, rbrev=cstrev.size()-1;
	                backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
	                denominator +=calculate_denominator_rev(patrev,lbrev,rbrev);
		}
	}else{}
	return denominator;
}

double pkn(cst_sct3<csa_sada_int<>>::string_type pat,bool mkn)
{
        int size = pat.size();
	double probability=0;
	if(size==ngramsize && ngramsize!=1){//for the highest order ngram
		int c=0;
		uint64_t lb=0, rb=cst.size()-1;
        	backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
//	        c = rb-lb+1;
		c = count(cst,pat);
		double D=0;
		/////// Different Discounts for mKN and KN////////////////
		if(mkn)
		{
			if(c==1){
				if(n1[size]!=0)	
					D = D1[size];
			}else if(c==2){
				if(n2[size]!=0)
					D = D2[size] ;
			}else if(c>=3){
				if(n3[size]!=0)
					D = D3[size];
			}
		}
	        else{
			D=Y[size]; 
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
		cst_sct3<csa_sada_int<>>::string_type pat2(size-1);
		for(int i=1;i<size;i++) //pat: a b c -> pat2: b c
		{
			pat2[i-1] = pat[i];			
		}
		/////// The same denominator for both KN and mKN /////
                pat.resize(size-1);
		denominator = count(cst,pat);
		if(denominator==0)
	        {
			cout<<pat<<endl;
        	        cout<<"---- Undefined fractional number XXXZ ----"<<endl;
//                	return 0;
			return (0+pow(2,(-10)));//TODO fix this!
		}
                lb=0;rb=cst.size()-1;
                backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
                auto v = cst.node(lb,rb);
		if(count(cst,pat)>0)
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
			uint64_t lbtemp=0, rbtemp=cst.size()-1;
	                backward_search(cst.csa, lbtemp, rbtemp, pat.begin(), pat.end(), lbtemp, rbtemp);
        	        auto vtemp = cst.node(lbtemp,rbtemp);
			int N1=0,N2=0,N3=0;
			if(count(cst,pat)>0)
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
							pat.resize(pat.size()+1);
							pat[pat.size()-1]=symbol;
		
							if(count(cst,pat)==1)
			                	                N1+=1;
			        	                else if(count(cst,pat)==2)
				                                N2+=1;
				                        else if(count(cst,pat)>=3)
		                              			N3+=1;

							pat.resize(pat.size()-1);
						}					
						++ind;
					}	
				}else{
		 			 int symbol = cst.edge(vtemp,pat.size()+1);
					 if(symbol!=1&&symbol!=0)
		     	                 {
					 	pat.resize(pat.size()+1);
						pat[pat.size()-1]=symbol;

						if(count(cst,pat)==1)
							N1+=1;
						else if(count(cst,pat)==2)
							N2+=1;
						else if(count(cst,pat)>=3)
							N3+=1;

						pat.resize(pat.size()-1);
		      		         }
				}
			}else{

			}
                        gamma = (D1[size] * N1) + (D2[size] * N2) + (D3[size] * N3);
			double output = (numerator/denominator) + (gamma/denominator)*pkn(pat2,mkn);
/*
			cout<<"D1 "<<D1[size]<<" D2 "<<D2[size]<<" D3 "<<D3[size]<<" N1 "<<N1<<" N2 "<<N2<<" N3 "<<N3<<" numerator is: "<<numerator<<" denomiator is: "<<denominator<<endl;
			cout<<"gamma "<<gamma<<endl;
			cout<<"HIGHESTGRAM"<<endl;
  	 		cout<<"output "<<output<<endl;
*/
                        return output; 

		}
		else{
//		 	cout<<"N is: "<<N<<" D is: "<<D<<" numerator is: "<<numerator<<" denomiator is: "<<denominator<<endl;	
  //           		cout<<"------------------------"<<endl;
			double gamma = D*N;
        	        return (numerator/denominator) + (gamma/denominator)*pkn(pat2,mkn);
		}
	} else if(size<ngramsize&&size!=1){//for lower order ngrams
		int c=0;         
		cst_sct3<csa_sada_int<>>::string_type patrev(pat.size());
		for(int i=0;i<pat.size();i++)// pat: a b c -> patrev: c b a
		{
			patrev[i]=pat[pat.size()-1-i];
		}
                uint64_t lbrev=0, rbrev=cstrev.size()-1;
                backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
                auto vrev = cstrev.node(lbrev,rbrev);
		if(count(cstrev,patrev)>0)
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
                        if(count(cst,pat)==1){
				if(n1[size]!=0)
					D = D1[size];
                        }else if(count(cst,pat)==2){
				if(n2[size]!=0)
					D = D2[size];
                        }else if(count(cst,pat)>=3){
				if(n3[size]!=0)
					D = D3[size];
			}
                }
                else{
                        D=Y[size];                                            
                }

	        double numerator=0;
        	if(c-D>0)
	        {
                	numerator = c-D;
        	}

		cst_sct3<csa_sada_int<>>::string_type pat3(size-1);
                for(int i=1;i<size;i++) //pat: a b c -> pat3: b c
                {
                        pat3[i-1] = pat[i];
                }
	
                pat.resize(pat.size()-1);// pat: a b c -> pat: a b
		uint64_t lb=0, rb=cst.size()-1;
                backward_search(cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
                auto v = cst.node(lb,rb);
	        int N=0;
		if(count(cst,pat)>0)
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
		double denominator = calculate_denominator(pat,lb,rb);
                if(denominator==0)
                {
                        cout<<"---- Undefined fractional number XXXW----"<<endl;
                        return 0;
                }		

		if(mkn)
		{
			 double gamma = 0;
                         uint64_t lbtemp=0, rbtemp=cst.size()-1;
                         backward_search(cst.csa, lbtemp, rbtemp, pat.begin(), pat.end(), lbtemp, rbtemp);
                         auto vtemp = cst.node(lbtemp,rbtemp);
                         int N1=0,N2=0,N3=0;
			 if(count(cst,pat)>0)
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
			                                pat.resize(pat.size()+1);
			                                pat[pat.size()-1]=symbol;

			                                if(count(cst,pat)==1)
			                                        N1+=1;
			                                else if(count(cst,pat)==2)
			                                        N2+=1;
			                                else if(count(cst,pat)>=3)
			                                        N3+=1;
					
			                                pat.resize(pat.size()-1);
						}
		                                ++ind;
		                        }
		                 }else{
					int symbol = cst.edge(vtemp,pat.size()+1);
					if(symbol!=1&&symbol!=0)
		     	                {
			                        pat.resize(pat.size()+1);
			                        pat[pat.size()-1]=symbol;

			                        if(count(cst,pat)==1)
			                                N1+=1;
			                        else if(count(cst,pat)==2)
			                                N2+=1;
			                        else if(count(cst,pat)>=3)
			                                N3+=1;

			                        pat.resize(pat.size()-1);
			                }
		                 }
			 }else{
			 }
                         gamma = (D1[size] * N1) + (D2[size] * N2) + (D3[size] * N3);
			 double output = numerator/denominator + (gamma/denominator)*pkn(pat3,mkn);
/*
			 cout<<"D1 "<<D1[size]<<" D2 "<<D2[size]<<" D3 "<<D3[size]<<" N1 "<<N1<<" N2 "<<N2<<" N3 "<<N3<<" numerator is: "<<numerator<<" denomiator is: "<<denominator<<endl;
			 cout<<"gamma "<<gamma<<endl;
    	                 cout<<"HIGHERGRAM"<<endl;
			 cout<<"output "<<output<<endl;
*/
			 return output;
		}
                else{

//		        cout<<"resized pat is: "<<pat<<" N is: "<<N<<" D is: "<<D<<" numerator is: "<<numerator<<" denomiator is: "<<denominator<<endl; 
//                        cout<<"------------------------"<<endl;
			double gamma = D*N;
			return (numerator/denominator) + (gamma/denominator)*pkn(pat3,mkn);
		}
	}

	else if(size==1 || ngramsize == 1)//for unigram
	{
		double D=0;
		////// Different Discounts for mKN and KN////////////////
                if(mkn)
                {
                        if(count(cst,pat)==1){
                                if(n1[size]!=0)
					D = D1[size];
                        }else if(count(cst,pat)==2){
				if(n2[size]!=0)
					D = D2[size];
                        }else if(count(cst,pat)>=3){
				if(n3[size]!=0)
					D = D3[size];
			}
                }
                else{
                        D=Y[size];                                            
                }
		////////////////////////////////////////////////////////////
		int c=0;     
		cst_sct3<csa_sada_int<>>::string_type patrev(pat.size());
		for(int i=0;i<pat.size();i++)// pat: a b c -> patrev: c b a
		{
			patrev[i]=pat[pat.size()-1-i];
		}
                uint64_t lbrev=0, rbrev=cstrev.size()-1;
                backward_search(cstrev.csa, lbrev, rbrev, patrev.begin(), patrev.end(), lbrev, rbrev);
                auto vrev = cstrev.node(lbrev,rbrev);
		if(count(cstrev,patrev)>0)
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
				int symbol = cst.edge(vrev,patrev.size()+1);
				if(symbol!=1&&symbol!=0)
				{
			                c=1;
				}          
		        }
		}else{
		}
		double numerator=0;
                if(c-D>0)
                {
                       numerator = c-D;
                }

        	double denominator=unigramdenominator;

                if(denominator==0)
                {
                        cout<<"---- Undefined fractional number XXXXY ----"<<endl;
                        return 0;
                }

                pat.resize(pat.size()-1);// pat: a -> pat: -

		///// The gamma for mKN //////////////////////////////
		if(mkn)
		{
			double gamma = 0;
			int N1=0,N2=0,N3=0;
            		cst_sct3<csa_sada_int<>>::string_type pattemp(1);
			int ind=0;
			while(ind<cst.degree(cst.root()))
                	{
                     		auto w = cst.select_child(cst.root(),ind+1);
				int word = cst.edge(w,1);
		                if(word!=1&&word!=0)
		                {
					pattemp[0]=word;
					if(count(cst,pattemp)==1)
                                                N1+=1;
                                        else if(count(cst,pattemp)==2)
                                                N2+=1;
                                        else if(count(cst,pattemp)>=3)
                                                N3+=1;
					
                                }
			     ++ind;
			}                      

                         gamma = (D1[size] * N1) + (D2[size] * N2) + (D3[size] * N3);
		         double output = numerator/denominator + (gamma/denominator)* (1/(double)vocabsize);
/*
			 cout<<"D1 "<<D1[size]<<" D2 "<<D2[size]<<" D3 "<<D3[size]<<" N1 "<<N1<<" N2 "<<N2<<" N3 "<<N3<<" numerator is: "<<numerator<<" denomiator is: "<<denominator<<endl;
 			 cout<<"gamma "<<gamma<<endl;
			 cout<<"UNIGRAM"<<endl;
			 cout<<"output "<<output<<endl;
*/
			 return output;
		} else
		{
/*
        	        cout<<"pat is: "<<pat<<" numerator is: "<<numerator<<" D is: "<<D<<" N1+ is: "<<c<<" denomiator is: "<<denominator<<endl;
	                cout<<"------------------------"<<endl;
*/
			return (numerator/denominator) +  ((D*vocabsize)/denominator) * (1/(double)vocabsize);
		}
	}
	return probability;
}

int main(int argc,char *argv[])
{
		string path = argv[1]; // query file path
		bool ismkn = false; // ismKN
		if(((string)argv[2]).compare("true"))
			ismkn = false;
		else
			ismkn=true;
		ngramsize = stoi((string)argv[3]);
		cache_config configorig;
	
		construct(cst, "file.sdsl",configorig,0); // loading from a stored original file

	        cout << "cst in MiB : " << size_in_mega_bytes(cst) << endl;


		cache_config configrev;
                construct(cstrev,"filereversed.sdsl",configrev,0); //loading from a stored reversed file


	        cout << "cst rev in MiB : " << size_in_mega_bytes(cstrev) << endl;
                cout<<"*************************************************"<<endl;

		vocabsize = cst.degree(cst.root())-2;// -2 is for deducting the count for 0, and 1
//		cout<<"Vocab size: "<<vocabsize<<endl;
		
                clock_t start_prec = clock();

		n1.resize(ngramsize+1);
		n2.resize(ngramsize+1);
		n3.resize(ngramsize+1);
		n4.resize(ngramsize+1);

		//precompute n1,n2,n3,n4
		cst_sct3<csa_sada_int<>>::string_type pat(1);
		ncomputer(ngramsize, pat, 0);

		Y.resize(ngramsize+1);
		D1.resize(ngramsize+1);
		D2.resize(ngramsize+1);
		D3.resize(ngramsize+1);
		//compute discounts Y, D1, D2, D+3
		for(int size=1;size<=ngramsize;size++)
		{
                        Y[size]=(double)n1[size]/(n1[size]+2*n2[size]);
                        if(n1[size]!=0)
                               D1[size] = 1 - 2 * Y[size] * (double)n2[size]/n1[size];
                        if(n2[size]!=0)
                               D2[size] = 2 - 3 * Y[size] * (double)n3[size]/n2[size];
                        if(n3[size]!=0)
                               D3[size] = 3 - 4 * Y[size] * (double)n4[size]/n3[size];
		}

		clock_t end_prec = clock();

                double elapsed_secs_prec = double(end_prec - start_prec) / CLOCKS_PER_SEC;
                cout<<"*************************************************"<<endl;
	        cout << "Precomputing Ds and ns - Time in seconds: "<<elapsed_secs_prec<<endl;
                cout<<"*************************************************"<<endl;
               
		cout<<"------------------------------------------------"<<endl;
                cout<<n1<<endl;
                cout<<n2<<endl;
                cout<<n3<<endl;
		cout<<n4<<endl;
		cout<<"------------------------------------------------"<<endl;
	        for(int size=1;size<=ngramsize;size++)
                {
          	      cout<<"n= "<<size<<" Y= "<<Y[size]<<" D1= "<<D1[size]<<" D2= "<<D2[size]<<" D3= "<<D3[size]<<endl;
                }

		cout<<"N+1(..)= "<<unigramdenominator<<endl;
		cout<<"-------------------------------"<<endl;


		ifstream file(path);
		string line;
		double perplexity=0;
		int M = 0; // number of words in the test data

		start_prec = clock();
		while (getline(file,line))
		{
			double sentenceprobability=0;
			vector<int> vec(1);
			istringstream iss(line);
			string word;
			int size=0;
			while (std::getline(iss, word, ' '))
			{  		
				try{
				   vec.resize(size+1);
				   int num = stoi(word);
				   if(num<=vocabsize)// TODO ignores Out-Of-Vocab words
				   {
				 	   vec[size]=num;
					   ++size;
				   }
				}catch(std::out_of_range& e){
				}
				M++;
			}
			int END=0;
			int START=0;
			while(size>0 && END<size)
			{
				START = END-ngramsize+1;
				if(START<=0){START=0;}
				cst_sct3<csa_sada_int<>>::string_type pattern(END-START+1);
				int j=0;		
				for(int i = START;i<=END;i++)
				{
					pattern[j]=vec[i];
					++j;
				}
//     			cout<<pattern<<endl;
				sentenceprobability = sentenceprobability+log2(pkn(pattern,ismkn));
//				cout<<"sentenceprobability is: "<<sentenceprobability<<endl;
				++END;
			}
//			cout<<"++++++++++++++++++SENTENCE END+++++++++++++++++++"<<endl;
			perplexity+=sentenceprobability;
		}
		end_prec = clock();
                elapsed_secs_prec = double(end_prec - start_prec) / CLOCKS_PER_SEC;
                cout << "Query time - Time in seconds: "<<elapsed_secs_prec<<endl;

//		cout<<"M is: "<<M<<endl;
		perplexity = (1/(double)M) * perplexity;
		perplexity = pow(2,(-perplexity));
		cout<<"*************************************************"<<endl;
		if(ismkn)
			cout<<"Modified Kneser-Ney"<<endl;
		else
			cout<<"Kneser-Ney"<<endl;
		cout<<"Perplexity is: "<<perplexity<<endl;
		cout<<"*************************************************"<<endl;

}
