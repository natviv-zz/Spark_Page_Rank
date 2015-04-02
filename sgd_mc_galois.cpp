#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "LoneStar/BoilerPlate.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

typedef struct Node{
//Can belong to the Users or Movies matrix with size 1*k(low rank)
//Row number
int id;
std::vector<double> col_vec;
}Node;

typedef Galois::Graph::LC_CSR_Graph<Node, double> Graph;
typedef Graph::GraphNode GNode;
typedef std::pair<int, int> mypair; 

bool mycompare(const mypair& l, const mypair& r) { 
return l.second > r.second; 
}

void sgd_update(std::vector<double>& u,std::vector<double>& m,double rating,double eta, double lambda){
	//Implemenet u = u - eta*lam*u + eta*(A_ij-u'm)*m 
	double r = rating - vector_multiply(u,m);
	double factor = 1-eta*lambda;
	scalar_multiply(u,factor);
	factor = eta*r;
	scalar_multiply(m,factor);
	vector_add(u,m);
	scalar_multiply(m,1/factor);

}

void generateRandom(std::vector<double>& random_vector) {
int n = random_vector.size();
for(int i=0; i<n; ++i) {
random_vector[i] = rand()/(double)RAND_MAX;
}
}

void vector_add(std::vector<double>& u,std::vector<double>& m){
	for(int i=0;i<u.size();i++){
		u[i]+=m[i];
	}
}

void scalar_multiply(std::vector<double>& vec,double u){
for(int i=0;i<vec.size();i++)
	vec[i] *= u; 
}

double vector_multiply(std::vector<double>&x,std::vector<double>&y){

	double res = 0.0;
	for(int i=0; i<x.size();i++)
		res+=x[i]*y[i];
	return res;

}

int main(int argc,char** argv){
Galois::StatManager statManager
Graph g_train;
Graph g_test;

if (argc !=5)
{
std::cout<<"Invalid input"<<std::endl;
return 1;
}
int rank = atoi(argv[1]);
double lambda = atof(argv[2]);
Galois::setActiveThreads(atoi(argv[3]));
double eta = atof(argv[4]);
int num_rows;
int num_cols;
int entries_train;
int entries_test;
std::string trainfile;
std::string testfile;
std::string datadir (argv[5]);
std::string meta;
meta = datadir + "/meta";
std::ifstream input(meta.c_str());

if (!input)
{
std::cout<<"Failed to open file "<<meta<<std::endl;
}

std::cout<<"Opened Meta File"<<std::endl;
input>>num_rows>>num_cols;
input>>entries_train>>trainfile;
input>>entries_test>>testfile;
trainfile = datadir + "/" +trainfile;
testfile = datadir + "/" + testfile;
input.close();
std::map<mypair,double> test_matrix;
input.open(testfile.c_str());
iff(!input)
std::cout<<"Failed to open test file"<<testfile<<std::endl;
std::cout<<"Opened test file "<<testfile<<std::endl;
for (std::string line; getline(input>>r>>c>>val,line);){
	mypair p(r-1,c-1);
	test_matrix.insert(std::make_pair(p,val));
}


Galois::Graph::readGraph(g_train,trainfile);
Galois::Graph::readGraph(g_test,testfile);

Galois::do_all(g_train.begin(), g_train.end(), [&](GNode n) { 
Node& data = g.getData(n);
vector<double> rand_vec(rank); 
data.col_vec = generateRandom(rand_vec);
});

Galois::StatTimer SGDTime("SGD"); 
SGDTime.start();

//Number of iterations = 10; 
for(int i=0;i<10;i++){

//Operator instantiated in an anonymous fashion
double train_rmse = 0.0;
Galois::for_each(g_train.begin(),g_train.end(),[&](Node u, Galois::UserContext<Node>&ctx) {
for (auto e:g.out_edges(u))
	//u,m,R_ij,eta,lambda
	sgd_update(u.col_vec,g.getEdgeDst(e).col_vec,g.getEdgeData(e),eta,lambda);
	train_rmse + = pow(vector_multiply(u.col_vec,g.getEdgeDst(e).col_vec)-g.getEdgeData(e),2);
});

	std::cout<<"Train RMSE"<<train_rmse/entries_test<<std::endl;

	//Check Test RMSE;
	double test_rmse = 0.0;
Galois::for_each(g_train.begin(),g_train.end(),[&](Node u, Galois::UserContext<Node>&ctx) {
for (auto e:g.out_edges(u))
	//u,m,R_ij,eta,lambda
	int r = u.id;
	int c = g.getEdgeDst(e).id;
	if(test_matrix.find(make_pair(r,c))!=test_matrix.end())
	test_rmse + = pow(vector_multiply(u.col_vec,g.getEdgeDst(e).col_vec)-test_matrix.find(make_pair(r,c)),2);
});
	std::cout<<"Test RMSE"<<test_rmse/entries/test;

}

return 0;
}
