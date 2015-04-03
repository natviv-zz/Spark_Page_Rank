#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <Eigen/SparseCore>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#define EIGEN_DONT_PARALLELIZE
Eigen::VectorXd multiply(Eigen::SparseMatrix <double> x, Eigen::VectorXd w, int num_threads){

        //std::cout<<"Called multiply"<<std::endl;
	Eigen::VectorXd result = Eigen::VectorXd::Zero(x.rows());
	omp_set_num_threads(num_threads);
        #pragma omp parallel for
	for(int k = 0; k < x.outerSize(); ++k){
	  for(Eigen::SparseMatrix<double>::InnerIterator it(x,k); it; ++it){
			result(it.row()) += it.value()*w(it.col());	
		}
	}	

	return result;
}

double obj_func(Eigen::SparseMatrix<double> x_hat, Eigen::VectorXd w, Eigen::VectorXi y, int C){
	double value = 0.0;
	value += (0.5*w.squaredNorm());
	
	for(int k = 0; k < x_hat.outerSize(); ++k){
		double loc = 0.0;
		for(Eigen::SparseMatrix<double>::InnerIterator it(x_hat,k); it; ++it){
			loc += it.value()*w(it.row());	
		}
		loc = loc*y(k);
		value += C*log(1+exp(-loc));
	}	
	
	return value;	

}

//Use x_hat as input
Eigen::VectorXd grad_func(Eigen::SparseMatrix<double> x_hat, Eigen::VectorXd w, Eigen::VectorXi y, int C){

	Eigen::VectorXd result = w;
	for(int k = 0; k < x_hat.outerSize(); ++k){
		double loc = 0.0;
		double value = 0.0;
		for(Eigen::SparseMatrix<double>::InnerIterator it(x_hat,k); it; ++it){
			loc += it.value()*w(it.row());	
		}

		loc = loc*y(k);
		value = C*(1/(1+exp(-loc))-1)*y(k);

			   for(Eigen::SparseMatrix<double>::InnerIterator it(x_hat,k); it; ++it){
			result(it.row()) += value*it.value();	
		}

			

	}	
	
	return result;	
	

}

Eigen::VectorXd conjugate_grad(Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> X_hat, Eigen::VectorXd w, Eigen::VectorXi y, Eigen::VectorXd b, int C, int num_threads){
		
                //std::cout<<"Called conjugate "<<std::endl;
		Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
		Eigen::VectorXd r = b;
		Eigen::VectorXd p = r;
		double alpha = 0.0;
		double beta = 0.0;
		Eigen::VectorXd old_r = b;
		while(1){
			//Compute temp = Xv nx1
			//std::cout<<"Entered loop "<<std::endl;
			Eigen::VectorXd temp = multiply(X,p,num_threads);
			//Compute temp' = Dtemp nx1
			for(int k = 0; k < X_hat.outerSize(); ++k){
				double loc = 0.0;
				for(Eigen::SparseMatrix<double>::InnerIterator it(X_hat,k); it; ++it){
					loc += it.value()*w(it.row());	
				}
				loc = loc*y(k);
				double sigma = 1/(1+exp(-loc));
				temp(k) *=  (sigma*(1-sigma));
			}
			// X'DX
			Eigen::VectorXd temp2 = multiply(X_hat,temp,num_threads);
			//Compute Ap = p + C*temp2

			alpha = r.dot(r)/p.dot(p+C*temp2);
			x = x + alpha*p;
			old_r = r;
			r = r - alpha*(p+C*temp2);
			if (r.squaredNorm()/b.squaredNorm() <= 0.01)
				break;
			//std::cout<<"Current squared norm is "<<r.squaredNorm()/b.squaredNorm()<<std::endl;
			beta = r.dot(r)/old_r.dot(old_r);
			p = r + beta*p;
			 								
		}
				
		return x;
}


int main(int argc, char* argv[])
{
  if (argc !=5)
    {
      std::cout<<"Invalid input"<<std::endl;
      return 1;
    }

  Eigen::initParallel();
  double C = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  std::cout<<"Threads "<<num_threads<<std::endl;
  int num_rows;
  int num_cols;
  int entries_train;
  int entries_test;
  std::string trainfile = argv[3];
  std::string testfile = argv[4];
  //std::cout<<trainfile<<std::endl;
  if (trainfile.compare("covtype.tr")==0) {
    entries_train = 500000;
    entries_test = 81012;
    num_cols = 54;
    num_rows = entries_train;
    std::cout<<"Matched covtype.tr" <<std::endl;

  }
  else {
     
    entries_train = 677399;
    entries_test = 20242;
    num_cols = 47276;
    num_rows = entries_train;
    std::cout<<"Matched rcv1.tr" <<std::endl;
	
  }
  std::ifstream input;

  Eigen::SparseMatrix <double> X(entries_train,num_cols);
  Eigen::SparseMatrix <double> Xtran(num_cols,entries_train);
  Eigen::SparseMatrix <double> Xt(entries_test,num_cols);
  Eigen::VectorXi Y(entries_train);
  Eigen::VectorXi Yt(entries_test);
  //std::cout<<"Entries train "<<entries_train<<std::endl;

  typedef Eigen::Triplet<double> T;
  std::vector<T> train_list,test_list,trans_list;
  //train_list.reserve(entries_train*num_cols);
  //test_list.reserve(entries_test*num_cols);
  //trans_list.reserve(entries_train*num_cols);
  int r,c,i;
  float val;
  std::string spaceDelimit = " ";
  std::string delimiter = ":";
  input.open(trainfile.c_str());
  if(!input)
    std::cout<<"Failed to open training file "<<trainfile<<std::endl;

  //std::cout<<"Opened training file "<<trainfile<<std::endl;
  r=0;
  double rel_error_array[20];
  double wall_time[20];
  while (!input.eof())
    {
      std::string line;
      getline(input,line);
      std::size_t pos = 0;
      i = 0;
      while ((pos = line.find(spaceDelimit)) != std::string::npos)
      {
	if(i==0)
	{
	  int label = atoi(line.substr(0,pos).c_str());
	  Y(r) = label;
	  i++;
	}
	else
	{
	  std::string temp = line.substr(0,pos);
	  std::size_t pos1 = temp.find(delimiter);

	  c = atoi(temp.substr(0,pos1).c_str());
	  val = atof(temp.substr(pos1+1).c_str());
	  // std::cout<<"temp "<<temp<<" pos1 "<<pos1<<" val "<<val<<std::endl;
	  train_list.push_back(T(r,c-1,val));
	  trans_list.push_back(T(c-1,r,val));
	}

	line.erase(0,pos+1);
      }
      r++;
    }

  //std::cout<<r<<std::endl;
  X.setFromTriplets(train_list.begin(),train_list.end());
  Xtran.setFromTriplets(trans_list.begin(),trans_list.end());
  //std::cout<<"Finished reading train file"<<std::endl;
  input.close();
  //std::cout<<X.nonZeros()<<std::endl;
  //std::cout<<Xtran.nonZeros()<<std::endl;
  input.open(testfile.c_str());
  std::vector<T>().swap(train_list);
  if(!input)
    std::cout<<"Failed to open test file"<<testfile<<std::endl;

  r = 0;
  //std::cout<<"Opened test file "<<testfile<<std::endl;
  
  
  while (!input.eof())
    {
      std::string line;
      getline(input,line);
      std::size_t pos = 0;
      i = 0;
      while ((pos = line.find(spaceDelimit)) != std::string::npos)
      {
	if(i==0)
	{
	  int label = atoi(line.substr(0,pos).c_str());
	  Yt(r) = label;
	  i++;
	}
	else
	{
	  std::string temp = line.substr(0,pos);
	  std::size_t pos1 = temp.find(delimiter);
	  c = atoi(temp.substr(0,pos1).c_str());
	  val = atof(temp.substr(pos1+1).c_str());
	  // std::cout<<val<<std::endl;
	  test_list.push_back(T(r,c-1,val));
	}

	line.erase(0,pos+1);
      }
      r++;
    }

  Xt.setFromTriplets(test_list.begin(),test_list.end());
  //std::cout<<"Finished reading test file"<<std::endl;
  //std::cout<<r<<std::endl;
  std::vector<T>().swap(test_list);
  //std::cout<<Xt.nonZeros()<<std::endl;
  int d = num_cols;
  int n = entries_train;
  //Eigen::VectorXd alpha = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd weight = Eigen::VectorXd::Zero(d);

      //Check test accuracy
    Eigen :: VectorXd predictions(entries_test);
    predictions = multiply(Xt,weight,num_threads);
    double correct = 0;
    for (int l = 0; l<predictions.size();l++)
      {
	if(predictions[l]>0)
	  predictions[l] = 1;
	else
	  predictions[l] = -1;
	//std::cout<<l<<" "<<predictions[l]<<" "<<Yt[l]<<std::endl;
	if (Yt[l] == predictions[l])
	  correct++;
      }
    //std::cout<<correct<<" "<<entries_test<<std::endl;
    double accuracy = (correct/entries_test)*100;
    std::cout<<"Initial Prediction accuracy is "<<accuracy<<" % "<<std::endl;

  int max_iter = 10;
  double init = omp_get_wtime();
  for (int iter=0;iter<max_iter;iter++)
  {
    
    std::cout<<"Iteration number "<<iter+1<<std::endl;
    
    double start = omp_get_wtime();
    omp_set_num_threads(num_threads);
    //#pragma omp parallel for shared(weight) schedule(dynamic)
    //std::cout<<"Called gradient function"<<std::endl;
    Eigen::VectorXd grad = grad_func(Xtran,weight,Y,C);
    grad=grad*-1;	
    //std::cout<<"Called conjugate gradient"<<std::endl;
    Eigen::VectorXd d = conjugate_grad(X,Xtran,weight,Y,grad,C,num_threads);
    grad = grad*-1;
    double alpha = 1;
    int i = 0;				
    while(true){
	alpha = pow(2,-i);
	Eigen::VectorXd temp = weight + alpha*d;
	//std::cout<<"Called objective function"<<std::endl;
	if(obj_func(Xtran,temp,Y,C) - obj_func(Xtran,weight,Y,C) < 0.01*alpha*d.dot(grad)){
	  weight += alpha*d;
		break; 
	}
	i++;
    }		
	
    std::cout<<"Iteration time "<<omp_get_wtime()-start<<std::endl;
    std::cout<<"Objective function "<<obj_func(Xtran,weight,Y,C)<<std::endl;
    //Check test accuracy
    Eigen :: VectorXd predictions(entries_test);
    predictions = multiply(Xt,weight,num_threads);
    double correct = 0;
    for (int l = 0; l<predictions.size();l++)
      {
	if(predictions[l]>0)
	  predictions[l] = 1;
	else
	  predictions[l] = -1;
	//std::cout<<l<<" "<<predictions[l]<<" "<<Yt[l]<<std::endl;
	if (Yt[l] == predictions[l])
	  correct++;
      }
    //std::cout<<correct<<" "<<entries_train<<std::endl;
    double accuracy = (correct/entries_test)*100;
    std::cout<<"Prediction accuracy is "<<accuracy<<" % "<<std::endl;

    

    
    
  }
  //std::cout<<"Relative error "<<std::endl;
  //for(int i = 0; i<20;i++)
    //std::cout<<wall_time[i]<<" "<<rel_error_array[i]<<std::endl;
  
  std::cout<<"Total time "<<omp_get_wtime()-init<<std::endl;
  return 0;
}


