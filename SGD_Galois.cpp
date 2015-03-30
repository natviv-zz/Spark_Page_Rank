#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Galois/Runtime/WorkList.h"
#include "Lonestar/BoilerPlate.h"
#include<sys/time.h>
#include <utility>
#include <algorithm>
#include <iostream>
#include "Galois/Statistic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <cmath>
#include <vector>
#include <map>
#include <queue>
#include <sys/time.h>

using namespace std;
#define pair_int pair< int, int >

int n,d,src,dest,weight;
float neta;
float intercept = 0.0;

struct comp {
    bool operator() (const pair_int &a, const pair_int &b) {
        return a.second > b.second;
    }
};

struct node {
	map<int, double > features;
	vector<double> w;
	int id;
};

struct feature {
	double value;
	int id;
};

vector<double > Y;
vector<node > Graph;
vector<feature> w;

struct Process {
	Process(){}
	template<typename Context>
		void operator()(node& source, Context& ctx) {
			double val = 0.0 - Y[source.id];
            		for (map<int, double>::iterator it=Graph[source.id].features.begin(); it!=Graph[source.id].features.end(); ++it) {
				val = val + w[it->first].value * it->second;	
			}
            		for (map<int, double>::iterator it=Graph[source.id].features.begin(); it!=Graph[source.id].features.end(); ++it) {
				w[it->first].value -= (double)neta * it->second * val;
			}
		}
};

struct Process1 {
	Process1(){}
	template<typename Context>
		void operator()(feature& source, Context& ctx) {
                	double mean = 0;
                	for (int i = 0; i < Graph.size(); i++) {
                    		mean += Graph[i].w[source.id];
                	}
                	w[source.id].value = mean/Graph.size();
                }
};

int main(int argc, char* argv[]) {
	Galois::StatManager statManager;
	ifstream inFile;
	std::string line;
        neta = 0.0001; 
	int threads = 1;
        int iter = 100;
        int show_errors = 1;
	inFile.open("inputfile", ifstream::in);
        
        if(argc > 3) {
            neta = atof(argv[1]);
            iter = atoi(argv[2]);
	    threads = atoi(argv[3]);
            if(argc > 4) show_errors = atoi(argv[4]);
            if(argc > 5) filename = argv[5];
        }
	Galois::setActiveThreads(threads);
	inFile.open(filename, ifstream::in);
        if(!inFile.is_open())
        {
		cout << "Unable to open file graph.txt";
                return 0;
        }
    getline(inFile, line);
    istringstream iss(line);
    iss>>n>>d;
	int maxX = 0;
	Y.resize(n);
	Graph.resize(n);
	w.resize(d);
        double initial_w = 0;
	int j=0;
	while (j < n)
	{
            getline(inFile, line);
            istringstream iss(line);
            iss >> Y[j]; string k; int i = 0;
            Graph[j].w.resize(d);
            // for (int i=0; i<d; i++) {
            while(iss >> k) {
                // if(j == 0) w[i] = initial_w;
                // int k = 1;
                // inFile >> k; 
                if(strcmp(filename, "rcv1") == 0) {
                    size_t pos = k.find(":");
                    Graph[j].features[atoi((k.substr(0,pos)).c_str())] = atof((k.substr(pos+1)).c_str());
                    maxX = 255;
                }
                else {   
                    if(atoi(k.c_str()) != 0) Graph[j].features[i] = atoi(k.c_str());
                    if(abs(atoi(k.c_str())) > maxX) maxX = abs(atoi(k.c_str()));
                }
                i++;
                    }
                    j++;
	}
 
        //Normalize
	if(maxX != 0) {
            for(int j = 0; j < n; j++) {
                for(int i = 0; i< d; i++) 
                    if (Graph[j].features.find(i) != Graph[j].features.end()) Graph[j].features[i] /= maxX;
                Y[j] /= maxX;
            }
            cout<< "Factor :" << maxX << endl;
        }

	inFile.closse();
	
	
	typedef Galois::WorkList::ChunkedFIFO<32> WL;
	
    	struct timeval start, end;
    	gettimeofday(&start, NULL);

	for (int k = 0; k < iter; k++) {
 		Galois::for_each<WL>(Graph.begin(), Graph.end(), Process());
                if(show_errors > 0) {
                    double error = 0.0;
                    for (int i = 0; i < n; i++) {
                            double partError = 0.0 - Y[i];
                            for (std::map<int, double>::iterator it=Graph[i].features.begin(); it!=Graph[i].features.end(); ++it)
                                 partError += w[it->first].value * it->second;
                            error = error + partError * partError;
                    }
                    error = error * maxX * maxX / n;
                    cout<<"Error : "<<error<<endl;
                }
    	}
                        double error = 0.0;
                    for (int i = 0; i < n; i++) {
                            double partError = 0.0 - Y[i];
                            for (std::map<int, double>::iterator it=Graph[i].features.begin(); it!=Graph[i].features.end(); ++it)
                                 partError += w[it->first].value * it->second;
                            error = error + partError * partError;
                    }
                    error = error * maxX * maxX / n;
                    cout<<"Error : "<<error<<endl;
                    
	cout << "SGD Completed" << endl;
	
  	return 0;
}
