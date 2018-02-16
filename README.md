# one-off-problems
Attempts at solving one off problems where dataset is available in public or can be mined 


1) FASHION MNIST PROBLEM
    - File: fashion_mnist_net.py
    - Running this file trains a Convolutional Neural Network.
    - Command line arguments are -e number_of_epochs and -p path_to_save_file 
    - Code written using PyTorch and can be run on GPU
    - This CNN accepts images of dimensions 28x28x1
    - This CNN has first layer as a conv layer with 16 filters, window size 5x5 and padding=2. This layer converts 28x28x1 input to 28x28x16 output
    - The second layer is a max pool layer of window 2x2. This layer converts 28x28x16 input to a 14x14x16 output
    - The third layer is a conv layer with 32 filters and window size 5x5. This layer converts a 14x14x16 input to 14x14x32
    - The fourth layer is a max pool layer of window 2x2. This layer converts 10x10x16 input to a 7x7x32 output
    - The 5th, 6th and 7th layers are fully connected layers that reduce dimensionality from 32*7*7 to 160, 84 and 10 respectively
    - The code trains a network with above architecture and returns the test accuracy
    - Currently, this architecture gives an accuracy of 91% on test set

![Alt text](images/tensorboard.png?raw=true "Accuray")
