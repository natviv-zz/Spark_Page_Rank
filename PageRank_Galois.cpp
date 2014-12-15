#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include <iostream>
#include <fstream>

#include <cstdio>
#include <cstdlib>

using namespace std;

typedef Galois::Graph::LC_CSR_Graph<int, void> Graph;
typedef Graph::GraphNode GNode;

Graph A;

struct Multiply {
    std::vector<double>& x;
    std::vector<double>& res;
    std::vector<double>& D;
    double constant;

    Multiply(std::vector<double>& x, std::vector<double>& res, std::vector<double> &D, double& constant) : x(x), res(res), D(D), constant(constant) {};

    void operator()(GNode& src, Galois::UserContext<GNode>& ctx) {
        int srcIndex = A.getData(src);

        res[srcIndex] = 0.0;
        for (auto edge : A.out_edges(src)) {
            GNode dst = A.getEdgeDst(edge);
            int dstIndex = A.getData(dst);

            res[srcIndex] += x[dstIndex];
        }

        res[srcIndex] = D[srcIndex] * res[srcIndex] + constant;
    }
};

void generateRandom(std::vector<double>& random_vector) {
    int n = random_vector.size();

    for(int i=0; i<n; ++i) {
        random_vector[i] = rand()/(double)RAND_MAX;
    }
}

void createGraph(const char*filePath, int nRows, std::vector<double>&x, std::vector<double>& D) {
    std::ifstream input;
    input.open(filePath);
    string line, row, col;
    double value;
    std::cout<<"Came here in CreateGraph"<<std::endl;
    for(int cIndex = 0; cIndex < nRows; ++cIndex) {
        getline(input >> row >> col >> value, line);
        int r = atoi(row.c_str())-1, c = atoi(col.c_str())-1;
        D[r]++;
    }
    input.close();
    std::cout<<"Mach 2"<<std::endl;
    for(int i = 0; i < 1770961; ++i) {
        D[i] = 0.85 / D[i];
        x[i] = 1 / 1770961;
    }
}

int main(int argc, char** argv) {
    Galois::setActiveThreads(atoi(argv[1]));

    Galois::Graph::readGraph(A, "/scratch/02681/natviv/friendster.gr");

    int dim = 15851028;
    std::vector<double> x, res, D;
    x.resize(dim);
    res.resize(dim);
    D.resize(dim);
    std::cout<<"Calling Create graph"<<std::endl;
    //createGraph("/scratch/02681/natviv/livejournal1.gr", 83663478, x, D);

    int id = 0;
    for (Graph::iterator ii = A.begin(), ei = A.end(); ii != ei; ii++) {
        A.getData(*ii) = id++;
    }

    std::cout<<"Mach 3"<<std::endl;
    double constant = 0.15 / dim;

    double value = 0;
    //Galois::StatTimer T
    for(int i = 0; i < 10; ++i) {
        Galois::StatTimer T;
        T.start();
        Galois::for_each(A.begin(), A.end(), Multiply(x, res, D, constant));

        x = res;
        T.stop();

        value += T.get();
    }

    std::cout<< "Page rank time "<<value<<std::endl;
    std::cout<< "Average value : " << value / 1000 << std::endl;

    return 0;
}
