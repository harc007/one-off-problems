# one-off-problems
Attempts at solving one off problems where dataset is available in public or can be mined 


1) FASHION MNIST PROBLEM
    - File: fashion_mnist_net.py
    - Running this file trains a Convolutional Neural Network.
    - Command line arguments are -e number_of_epochs and -p path_to_save_file 
    - Code written using PyTorch and can be run on GPU
    - This CNN accepts images of dimensions 28x28x1
    - This CNN has first layer as a conv layer with 6 filters and window size 5x5. This layer converts 28x28x1 input to 24x24x6 output
    - The second layer is a max pool layer of window 2x2. This layer converts 24x24x6 input to a 12x12x6 output
    - The third layer is a conv layer with 16 filters and window size 3x3. This layer converts a 12x12x6 input to 10x10x16
    - The fourth layer is a max pool layer of window 2x2. This layer converts 10x10x16 input to a 5x5x16 output
    - The fifth layer is a conv layer with 24 filters and window size 3x3. This layer converts a 10x10x16 input to 8x8x24
    - The 6th, 7th and 8th layers are fully connected layers that reduce dimensionality from 16*4*4 to 120, 84 and 10 respectively
    - The code trains a network with above architecture and returns the test accuracy
    - Currently, this architecture gives an accuracy of 89% on test set
