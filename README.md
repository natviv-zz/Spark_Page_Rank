Spark_Page_Rank
===============
This repository contains implementation of PageRank and SGD algorithm for Matrix Factorization on Galois. It also has the code used to run PageRank and SGD using IPM appraoch on Spark.

For Mahout and Graphlab, used the implementations already present for comparisons.

Page Rank

D_PageRank_Galois.cpp - Code corresponding to Galois Page Rank in Figure 4. The different opeartors and schedulers have been specified for the different appraoches. (Dynamic is essentailly running until convergence and not for a fixed number of iterations)

PageRank_Galois.cpp is an inefficient implementation of Page Rank in Galois

PageRank_OMP.cpp is the Page Rank implementation parallelized using OpenMP.

Dynamic_Page_Rank.scala and PageRank.scala - They are different Spark implementations of the Page Rank algorithm. I directly referred to this tutorial for running Page Rank on Spark. It makes use of the first implementation under the hood.
http://ampcamp.berkeley.edu/big-data-mini-course/graph-analytics-with-graphx.html for the comparison in Figure 1.

Pregel_PageRank.scala - A message passing based synchronous Page Rank appraoch in Spark. Used for experimentation. Not used in final report.

Matrix Completion:

sgd_mc_galois.cpp - It has the SGD algorithm for Matrix Completion in Galois. The same code was modified and used for ALS implementation of Galois in Figure 4 which is a modification of the opeartor and the schedule. I had also experimented with using Eigen library's data structures for solving the least squares problem.(Unfortunately, I did not commit the changes)

als_omp.cpp - Implementation of ALS algorithm parallelized using OpenMP. Used for reference purposes.  

Spark_ALS - Used MLLib's ALS implementation for comparison

Logistic Regression - 
logistic_regression.cpp - Contains Logistic Regression implementation parallelized using OpenMP

sgd_omp.cpp - A simple implementation of SGD parallelized using OpenMP.

SGD_Galois.cpp - Contains the SGD implementation in Galois used for comparisons in Figure 6

Spark DSGD - Used MLLib's implementation of SGD for comparison in Figure 6

IPM_SGD.py - The Spark implemenation of IPM used in Figure 6.











