'''
Here we use PyTorch Variables and autograd to implement our 2-layer network;
now we no longer need to manually implement the backward pass through the network:
'''

import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold input and outputs, and wrap them in Variables.
# setting requires_grad = Flase indicates that we do not need to compute gradients

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad = True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Variables: these are
    # exactlyu the same operations we used to compute the forward pass using 
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2) # foward prop

    # Compute and print loss using operations on Variables.
    # Now, loss is a Variable of shape (1, ) and loss.data is a Tensor of shape
    # (1,); loss.date[0] is a scalar value holding the loss.
    loss = (y_pred - y).pow(2).sum()
    print('t and loss...', t, loss.data[0], type(loss))

    # Use autograd to compute the backward pass. This call will compute the 
    # gradient of loss with respect to all Variables with requires_grad = True.
    # After this call w1.grad and w2.grad will be Variables holding the gradient 
    # of the loss with respect to w1 and w2 respectively.
    loss.backward() # Backprop automatically!

    # update weights using gradient descent: w1.data and w2.data are Tensors.
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are 
    # Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()
