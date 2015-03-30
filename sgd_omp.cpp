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
#include <omp.h>
#include <sstream>
#include <sstream>

using namespace std;
#define pair_int pair< int, int >

// struct node {
//  map<int, float > features;
//  float w;
// };

int main(int argc, char* argv[]) {
    ifstream inFile;
    std::string line;
    int n,d,src,dest,weight;
    float intercept = 0.0;
    float neta = 0.0001;
    int threads = 1;
    int iter = iter_default;
    char* filename = "rcv1";
    double lambda1 = 0.0;
    double lambda2 = 0.0;
    int show_errors = 1;
        
        if(argc > 3) {
            neta = atof(argv[1]);
            iter = atoi(argv[2]);
            threads = atoi(argv[3]);
            if(argc > 4) show_errors = atoi(argv[4]);
            if(argc > 5) filename = argv[5];
            if(argc > 6) {lambda1 = atof(argv[6]);lambda2 = atof(argv[7]);}
        }
    inFile.open(filename, ifstream::in);

    if(!inFile.is_open())
        {
        cout << "Unable to open file";
                return 0;
        }
    getline(inFile, line);
    istringstream iss(line);
    iss>>n>>d;
    vector<float> Y;
    vector<map<int, float> > X;
    vector<float> w;

    int maxX = 0;
    Y.resize(n);
    X.resize(n);
    w.resize(d);
    float initial_w = 0;
    int j=0;
    while (j < n)
    {
        getline(inFile, line);
        istringstream iss(line);
        iss >> Y[j]; string k; int i = 0;
        while(iss >> k) {
            if(strcmp(filename, "rcv1") == 0) {
                size_t pos = k.find(":");
                X[j][atoi((k.substr(0,pos)).c_str())] = atof((k.substr(pos+1)).c_str());
                maxX = 255;
            }
            else {   
                if(atoi(k.c_str()) != 0) X[j][i] = atoi(k.c_str());
                if(abs(atoi(k.c_str())) > maxX) maxX = abs(atoi(k.c_str()));
            }
            i++;
        }
        j++;
    }

    if (j != n) {
        cout << "File input error" << endl; return 0;
    }   
    if(maxX != 0) {
        for(int j = 0; j < n; j++) {
            for(int i = 0; i< d; i++) 
                X[j][i] /= maxX;
            Y[j] /= maxX;
        }
        cout<< "Factor :" << maxX << endl;
    }
        
    inFile.close();
    float w_next;
    struct timeval start, end;
    gettimeofday(&start, NULL);
  
    #pragma omp parallel for num_threads(threads)             
    for (int k = 0; k < iter; k++) {
                int j = rand() % n;
                float val = intercept - Y[j];
            for (map<int, float>::iterator it=X[j].begin(); it!=X[j].end(); ++it) {
                val += (w[it->first] * it->second);
            }
            for (map<int, float>::iterator it=X[j].begin(); it!=X[j].end(); ++it) {
                w[it->first] = w[it->first]*(1+ lambda1) - (float)neta * it->second * val + lambda2;
            }
    if(show_errors > 0) {
        float error = 0.0;
        #pragma omp parallel for reduction(+ : error) num_threads(threads)
        for (int j1 = 0; j1 < n; j1++) {
            float partError = intercept - Y[j1];
            for (std::map<int, float>::iterator it=X[j1].begin(); it!=X[j1].end(); ++it)
                partError += w[it->first] * it->second;
            error += partError * partError;
        }
        error = error * maxX * maxX / n;
        cout<<"Error : "<<error<<endl;    
    }
}
    
            float error = 0.0;
        #pragma omp parallel for reduction(+ : error) num_threads(threads)
        for (int j1 = 0; j1 < n; j1++) {
            float partError = intercept - Y[j1];
            for (std::map<int, float>::iterator it=X[j1].begin(); it!=X[j1].end(); ++it)
                partError += w[it->first] * it->second;
            error += partError * partError;
        }
        error = error * maxX * maxX / n;
        cout<<"Error : "<<error<<endl;  
        
    gettimeofday(&end, NULL); 
    cout << "SGD Done" << endl;
    return 0;
}
