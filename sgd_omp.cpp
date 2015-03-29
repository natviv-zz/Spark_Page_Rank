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
#define neta_default .0001
#define iter_default 10
#define thread_default 10
#define is_intercept false

// struct node {
//  map<int, float > features;
//  float w;
// };

int main(int argc, char* argv[]) {
    ifstream inFile;
    std::string line;
    int n,d,src,dest,weight;
    float intercept = 0.0;
    float neta = neta_default;
    int threads = thread_default;
    int iter = iter_default;
    char* filename = "madelon"; // "inputfile"; //"madelon";
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
        cout << "Unable to open file graph.txt. \nProgram terminating...\n";
                return 0;
        }
    getline(inFile, line);
    istringstream iss(line);
    iss>>n>>d;
    // cout<<n<<" "<<d<<endl;
    vector<float> Y;
    vector<map<int, float> > X;
    vector<float> w;

    int maxX = 0;
    Y.resize(n);
    X.resize(n);
    w.resize(d);
        // maxX.assign(d,0);
    float initial_w = 0;
    int j=0;
    //j -> sample, i -> feature
    while (j < n)
    {
        getline(inFile, line);
        istringstream iss(line);
        iss >> Y[j]; string k; int i = 0;
        // for (int i=0; i<d; i++) {
        while(iss >> k) {
            // if(j == 0) w[i] = initial_w;
            // int k = 1;
            // inFile >> k; 
            if(strcmp(filename, "mnist") == 0) {
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
    //Normalize
    if(maxX != 0) {
        for(int j = 0; j < n; j++) {
            for(int i = 0; i< d; i++) 
                X[j][i] /= maxX;
            Y[j] /= maxX;
        }
        cout<< "Factor :" << maxX << endl;
    }
        
    inFile.close();
    //converging values: MADELON "***"
    cout << "No .of samples=" << n << " No of features=" << d << endl;
    cout << "Neta : "<< neta << " Iterations : "<< iter << " Threads :"<< threads << endl;

    float w_next;
    struct timeval start, end;
    gettimeofday(&start, NULL); //start time of the actual algorithm
  
    #pragma omp parallel for num_threads(threads)             
    for (int k = 0; k < iter; k++) {
                int j = rand() % n;
                float val = intercept - Y[j];
            // #pragma omp parallel for reduction(+ : val) num_threads(threads)
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
    cout << "SGD Completed" << endl;
    printf ("Elasped time is %.4lf seconds.\n", (((end.tv_sec  - start.tv_sec) * 1000000u +  end.tv_usec - start.tv_usec) / 1.e6) );
    // if(is_intercept) cout << intercept << endl;
    // for (int i=0;i< w.size();i++) {
    //  cout << w[i] << endl;
    //    }
    
    return 0;
}
