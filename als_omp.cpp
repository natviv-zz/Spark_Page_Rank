#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <Eigen/SparseCore>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/LU>
int main(int argc, char* argv[])
{
if (argc !=5)
{
std::cout<<"Invalid input"<<std::endl;
return 1;
}
int rank = atoi(argv[1]);
double lambda = atof(argv[2]);
int num_threads = atoi(argv[3]);
int num_rows;
int num_cols;
int entries_train;
int entries_test;
std::string trainfile;
std::string testfile;
std::string datadir (argv[4]);
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
std::cout<<num_rows<<" "<<num_cols<<std::endl;
std::cout<<"Training file "<<trainfile<<" Num entries "<<entries_train<<std::endl;
std::cout<<"Testing file "<<testfile<<" Num entries "<<entries_test<<std::endl;
Eigen::SparseMatrix <double> R(num_rows,num_cols);
Eigen::SparseMatrix <double> Rt(num_rows,num_cols);
Eigen::SparseMatrix <double> Rtran(num_cols,num_rows);
typedef Eigen::Triplet<double> T;
std::vector<T> train_list,trans_list,test_list;
train_list.reserve(entries_train);
trans_list.reserve(entries_train);
test_list.reserve(entries_test);
int r,c,val;
input.close();
input.open(trainfile.c_str());
if(!input)
std::cout<<"Failed to open training file "<<trainfile<<std::endl;
std::cout<<"Opened training file "<<trainfile<<std::endl;
int count=0;
for (std::string line;getline(input>>r>>c>>val,line);)
{
//std::cout<<r<<" "<<c<<" "<<val<<std::endl;
//R.insert(r-1,c-1)=val;
//Rtran.insert(c-1,r-1)=val;
train_list.push_back(T(r-1,c-1,val));
trans_list.push_back(T(c-1,r-1,val));
count++;
}
std::cout<<count<<std::endl;
R.setFromTriplets(train_list.begin(),train_list.end());
Rtran.setFromTriplets(trans_list.begin(),trans_list.end());
std::cout<<"Finished reading train file"<<std::endl;
input.close();
input.open(testfile.c_str());
std::vector<T>().swap(train_list);
std::vector<T>().swap(trans_list);
if(!input)
std::cout<<"Failed to open test file"<<testfile<<std::endl;
count = 0;
std::cout<<"Opened test file "<<testfile<<std::endl;
for (std::string line; getline(input>>r>>c>>val,line);)
{
//Rt.insert(r-1,c-1)=val;
test_list.push_back(T(r-1,c-1,val));
count++;
}
Rt.setFromTriplets(test_list.begin(),test_list.end());
std::cout<<"Finished reading test file"<<std::endl;
std::cout<<count<<std::endl;
std::vector<T>().swap(test_list);
int k =rank;
int u = num_rows;
int m = num_cols;
Eigen::MatrixXd U = Eigen::MatrixXd::Random(k,u);
Eigen::MatrixXd M = Eigen::MatrixXd::Random(k,m);
//std::cout<<"Now allocating for res"<<std::endl;
//Eigen::SparseMatrix<double> Res(u,m);
//std::cout<<U<<std::endl;
//std::cout<<M<<std::endl;
Eigen::MatrixXd lamI;
int max_iter = 10;
double start = omp_get_wtime();
std::cout<<"start with nr_threads "<<num_threads<<std::endl;
for (int iter=0;iter<max_iter;iter++)
{
lamI = lambda*Eigen::MatrixXd::Identity(k,k);
omp_set_num_threads(num_threads);
//std::cout<<"Time at iter "<<omp_get_wtime()-start<<std::endl;
#pragma omp parallel for schedule (dynamic)
for (int i = 0;i<m;i++)
{
//Implement find(R(:,i))
//std::cout<<"Current M col "<<i<<std::endl;
Eigen::MatrixXd Usub(k,0);
Eigen::MatrixXd Rsub(0,1);
Eigen::MatrixXd denom;
Eigen::MatrixXd numer;
Eigen::MatrixXd soln(k,1);
for (Eigen::SparseMatrix<double>::InnerIterator it(R,i);it;++it)
{
//std::cout<<it.value()<<" "<<it.row()<<" "<<it.col()<<" "<<it.index()<<std::endl;
//std::cout<<i<<std::endl;
//std::cout<<it.col()<<" "<<it.row()<<std::endl;
int curr_col = Usub.cols();
int curr_row = Usub.rows();
Usub.conservativeResize(curr_row,curr_col+1);
//std::cout<<"Current Usub size "<<curr_col<<std::endl;
Usub.col(curr_col)=U.col(it.row());
//std::cout<<Usub<<std::endl;
curr_row = Rsub.rows();
Rsub.conservativeResize(curr_row+1,1);
Rsub(curr_row,0)=it.value();
}
//std::cout<<"U sub size "<<Usub.rows()<<" "<<Usub.cols()<<std::endl;
//std::cout<<"R sub size "<<Rsub.rows()<<" "<<Rsub.cols()<<std::endl;
//#pragma omp critical
denom = Usub*Rsub;
numer = Usub*Usub.transpose()+lamI;
soln = numer.colPivHouseholderQr().solve(denom);
M.col(i) = soln;
}
//std::cout<<"Time in M loop "<<omp_get_wtime()-start<<std::endl;
#pragma omp parallel for schedule(dynamic)
for (int j = 0;j<u;j++)
{
//Implementing find(R(i,:))
//std::cout<<"Current U col "<<j<<std::endl;
Eigen::MatrixXd Msub(k,0);
Eigen::MatrixXd Rsub(0,1);
Eigen::MatrixXd denom;
Eigen::MatrixXd numer;
Eigen::MatrixXd soln(k,1);
for (Eigen::SparseMatrix<double>::InnerIterator it(Rtran,j);it;++it)
{
//std::cout<<it.value()<<" "<<it.row()<<" "<<it.col()<<" "<<it.index()<<std::endl;
//std::cout<<j<<std::endl;
//std::cout<<it.col()<<" "<<it.row()<<std::endl;
int curr_row = Msub.rows();
int curr_col = Msub.cols();
Msub.conservativeResize(curr_row,curr_col+1);
Msub.col(curr_col)=M.col(it.row());
curr_row = Rsub.rows();
Rsub.conservativeResize(curr_row+1,1);
Rsub(curr_row,0)=it.value();
}
//std::cout<<"Time find sub matrix "<<omp_get_wtime()-start<<std::endl;
//std::cout<<"Msub size "<<Msub.rows()<<" "<<Msub.cols()<<std::endl;
//std::cout<<"Rsub size "<<Rsub.rows()<<" "<<Rsub.cols()<<std::endl;
//#pragma omp critical
denom = Msub*Rsub;
numer = Msub*Msub.transpose()+lamI;
soln = numer.colPivHouseholderQr().solve(denom);
U.col(j)=soln;
//std::cout<<omp_get_num_threads()<<std::endl;
}
//std::cout<<"Time in U loop "<<omp_get_wtime()-start<<std::endl;
//Res = U.transpose()*M.sparseView();
double rmse = -0.3;
//#pragma omp parallel for num_threads(num_threads)
for (int l = 0;l<Rt.outerSize();++l)
for(Eigen::SparseMatrix<double>::InnerIterator it(Rt,l);it;++it)
{
int row = it.row();
int col = it.col();
double res_val = (U.col(row).array()*M.col(col).array()).sum();
double diff = res_val-it.value();
rmse = rmse + pow(diff,2);
}
rmse = sqrt(rmse/entries_test);
//std::cout<<"Iteration number "<<iter<<std::endl;
std::cout<<"iter "<<iter+1<<" Walltime "<<omp_get_wtime()-start<<" RMSE "<<rmse<<std::endl;
}
//std::cout<<lamI;
/*
Eigen::MatrixXd mat(2,2);
mat(0,0) = 3;
mat(1,0) = 2.5;
mat(0,1) = -1;
mat(1,1) = mat(1,0)+mat(0,1);
std ::cout << mat <<std::endl;
mat.diagonal()<< -1,-2;
std::cout<< mat << std::endl;
*/
return 0;
}

    Status
    API
    Training
    Shop
    Blog
    About

