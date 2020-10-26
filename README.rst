Safe screening for sparse regression with the Kullback-Leibler divergence
=========================================================================

Author: Cassio F. Dantas

This is a **Matlab** code corresponds to the following paper:

[1] C. F. Dantas, E. Soubies and C. Févotte  “Safe screening for sparse regression with the Kullback-Leibler divergence,” submitted to ICASSP 2021.


.. contents::


-----------
Disclaimers
-----------

- CoD_KL_l1_update.cpp needs to be compiled (>>cd solvers/, mex CoD_KL_l1_update.cpp)
- Datasets are not provided and need to be downloaded by the user and placed in a subfolder ./datasets (20 News Groups, NIPS papers, Taste Profile and Encyclopedia). See file load_dataset.m for further instructions.
- Synthetic experiments can be performed directly.


-----------------
Files description
-----------------

- main_KL_screening_test.m : main script that launches all experiments. Simulation parameters can be set in this file. See for instance the variable exp_type, which allows to choose between a synthetic experiment and the different real datasets.
- load_dataset.m : loads dataset from folder ./datasets and places data in the correct variables.
- ./screening_tests/: contains the functions performing the proposed screening tests (as well as the required precalculations).
- ./solvers/: contains all tested solvers along with their corresponding version using screening:
	- CoD (coordinate descent): code based on the original solver available at http://www.cs.utexas.edu/~cjhsieh/nmf and proposed by the authors of « Cho-Jui Hsieh and Inderjit S. Dhillon, Fast Coordinate Descent Methods with Variable Selection for Non-Negative Matrix Factorization. KDD 2011 ». 
	- SPIRAL (proximal gradient descent): the provided implementation is based on the original solver available at http://drz.ac/code/spiraltap/ and proposed by the authors of « Zachary T. Harmany, Roummel F. Marcia, Rebecca M. Willett, This is SPIRAL-TAP: Sparse Poisson Intensity Reconstruction Algorithms -- Theory and Practice. IEEE Transactions on Image Processing 2011 » .
	- MM (multiplicative update): homemade MU standard solver - full Matlab implementation.

Code for generating some plots are given in the final lines of main_KL_screening_test.m script.
