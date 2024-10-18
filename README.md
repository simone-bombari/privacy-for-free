### This repository contains the implementation of the experiments contained in the paper "Privacy for Free in the Over-Parameterized Regime".

main_RF.py contains a basic code to run DP-GD on a random features model. It runs on cpu, uses numpy and does not parallelize the gradients computations. At the current state of the paper, it is used to plot the test loss of \theta^* in Figure 3.

main_RF_grid.py is almost identical, but it is simply ready to run a grid search on the hyper-parameter space, to obtain the last plot of Figure 3 (at the current stage of the paper).

main_RF_time_JAX.py is similar, it is used to run DP-GD on models with different sizes (first three panels of Figure 2 in the current stage of the paper). The code is written to run on gpu, and uses the JAX library.

utils_NN_JAX.py is a file containing utils functions to run the files all the main_NN_* files.

main_NN_GD_JAX.py runs (non private) gradient descent with different training set sizes. The results are used in Figure 2.

main_NN_JAX.py runs DP-GD with determined number of training samples and validation samples. We use this code for both Figure 1 and Figure 2. Notice that this file produces both validation and test accuracies. To plot our results, we will look at the number of training iterations T that maximizes the average validation accuracy, and we will use this fixed hyper-parameter to plot the test accuracy. We do not retrain on the full dataset, which motivates why our maximum value of n is 50000.

main_NN_grid_JAX.py works as the previous file, but is set to provide the plot in Figure 4, i.e. the validation accuracy over different hyper-parameter used in the private optimization.