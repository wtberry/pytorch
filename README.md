# Pytorch tutorials & practices

Tutorials for [PyTorch](https://pytorch.org/) deep leaning platform.

* Done on spring 2018, documented by git on Jan 2019
## python scripts
- 60_min_cnn.py: pytorch [60 min tutorial neural networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)
- cifar10_tutorial.py: torchvision package tutorial with cnn
- custom_autograd.py: creating and using custom autograd functions, using torch.autograd.Function 
- MNIST_nn.py: practice building a simple fully connected network using nn.Sequential, and applying it to MNIST
- nn_connected_ReLU.py: dynamic fully connected neural net, re-using one hidden layers random numbers of times (1~3) in 1 forward pass. 
- nn_custom_class.py: custom neural net using subclass
- numpy_nn.py & pytorch_nn.py: implementing simple nn from scratch. makes me appreciate libraries...

## jupyter notebook tutorials
- tensor_tutorial.ipynb: very basics of PyTorch, how tensor works and commonalities with NumPy array
- autograd_tutorial.ipynb: tutorial on autograd, Variable, .backward and Function
- neural_networks_tutorial.ipynb: covers basic steps of training a simple neural net, from model definition to updating weights.
- sequence_models_tutorial.ipynb: part of speech tagging using LSTM from nn module & also by creating custom LSTM subclass.
