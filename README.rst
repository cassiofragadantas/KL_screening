GAP Safe screening with local strong-concavity bounds
=====================================================

Author: Cassio F. Dantas

This is a **Matlab** code corresponds to the following paper:

[1] C. F. Dantas, E. Soubies and C. Févotte  “Expanding Boundaries of GAP Safe Screening,” submitted to JMLR 2021.

[2] C. F. Dantas, E. Soubies and C. Févotte  “Safe screening for sparse regression with the Kullback-Leibler divergence,” submitted to ICASSP 2021.

It includes three main simulation cases: Logistic Regression, beta=1.5 divergence and Kullback-Leibler divergence. The proposed technique is the first screening rule in the literature capable of addressing the two last cited cases (beta div and KL).

.. contents::


-----------
Disclaimers
-----------

- CoD_KL_l1_update.cpp needs to be compiled (run 'mex CoD_KL_l1_update.cpp -lmwblas' from inside the solvers folder). Two compiled versions have been provided, but might not work on your computer: .mexa64(compiled on a Linux 64-bit architecture) and .mexmaci64 (compiled on a Mac 64-bit).
- Datasets are not provided and need to be downloaded by the user and placed in a subfolder ./datasets (Leukemia, Urban hyperspectral image, NIPS papers, 20 News Groups, Taste Profile and Encyclopedia). See file load_dataset.m for further instructions.
- Synthetic experiments can be performed directly.


-----------------
Files description
-----------------

- main.m : main script that launches all experiments. Simulation parameters can be set in this file. See for instance the variables: problem_type, which allows to choose one of the three treated data-fidelity functions; exp_type, which allows to choose between a synthetic experiment and the different real datasets. According to the chosen type of problem, one of the three following scripts are called by main.m
	- run_LogReg_solvers, run_Beta_solvers, run_KL_solvers, respectively for the logistic, beta divergence and KL cases.
- load_dataset.m : loads dataset from folder ./datasets and places data in the correct variables.
- ./screening_tests/: contains the functions performing the proposed screening tests (as well as the required precalculations).
- ./solvers/: contains all tested solvers along with their corresponding version using screening:
	- Kullback-Leibler
		- CoD (coordinate descent): code based on the original solver available at http://www.cs.utexas.edu/~cjhsieh/nmf and proposed by the authors of « Cho-Jui Hsieh and Inderjit S. Dhillon, Fast Coordinate Descent Methods with Variable Selection for Non-Negative Matrix Factorization. KDD 2011 ». See files: CoD_KL_l1.m, CoD_KL_l1_GAPSafe.m and CoD_KL_l1_update.cpp.
		- SPIRAL (proximal gradient descent): the provided implementation is based on the original solver available at http://drz.ac/code/spiraltap/ and proposed by the authors of « Zachary T. Harmany, Roummel F. Marcia, Rebecca M. Willett, This is SPIRAL-TAP: Sparse Poisson Intensity Reconstruction Algorithms -- Theory and Practice. IEEE Transactions on Image Processing 2011 » . See files: SPIRAL.m and SPIRAL_GAPSafe.m.
		- MU (multiplicative update): homemade MU standard solver - full Matlab implementation. See files: KL_l1_MM.m and KL_l1_MM_GAPSafe.m.
	- Beta=1.5 divergence
		- MU (multiplicative update): homemade MU standard solver - full Matlab implementation. See files: Beta_l1_MM.m and Beta_l1_MM_GAPSafe.m.
	- Logistic Regression
		- CoD (coordinate descent): a prox-Newton step is performed for each coordinate sub-problem. Implementation based on that on `GAP Safe original code <https://github.com/EugeneNdiaye/Gap_Safe_Rules>`_ (cf. file cd_logreg_fast.pyx). See files: LogReg_CoD_l1.m LogReg_CoD_l1_GAPSafe.m.

Some plots are automatically generated, only if a single regularization value is chosen.
