Start with

python3 ./augment_data.py

to create a folder with the MNIST and CIFAR-10 datasets, with the layer of noise added.
This will create the attacks queries as well.

You can use the notebook check_data.ipynb to verify that everything is working properly.


Next run a command like

python3 ./main.py --i 0 --fmap 'rf' --dataset 'synthetic' --activation 'phi2'

to run one of the experiments in Figure 3. You can modify the activation, the feature map, and the dataset. The python script will authomatically save the files in respective directories.


Finally, you can run

python3 ./main_NN.py --i 0 --dataset 'MNIST' --net 'SCN' --save 'trial'

to run a training of a small convolutional neural network on the MNIST dataset, as in Figure1, subpplot 3.


To plot your data_points, you can use the notebook to_plot.ipynb
