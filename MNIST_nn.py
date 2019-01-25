'''
Practice creating NN, 2 hidden layers, applied on MNIST dataset using pyTorch
'''

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

### Importing the MNIST Dataset from pyTorch torchvision
# The output of torchvision datasets are PILImage images of range [0, 1]

transform = transforms.Compose(
        [transforms.ToTensor()]) 
        # maybe normalize too??

train_set = torchvision.datasets.MNIST(root='./MNIST_data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)
# num_workers = how many subprocess to use to load the data

test_set = torchvision.datasets.MNIST(root='./MNIST_data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

## Let's take a look at some of the images
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg))

# Get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Show images 
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
plt.show()


### Setting up parameters for NN
# is batch size, D_in is input dimension
# H is hidden dimension, D_out is output dimension
N, D_in, H, D_out = 1000, 784, 50, 10

##### Creating model using NN package
''' Using the nn package to define our model as a sequence of layers. 
nn.Sequencial is a Module which contains other Modules, and applies them in 
sequence to produce its output.. Each linear Module computes output from input
using a linear function, and holds internal variables for its weight, and bias.
'''

model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
        )

''' The nn package also contains definitions of popular loss functions; in this
case, we will use Mean Squared Error (MSE) as our loss function.
'''
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

''' Using optim package to define an Optimizer that will update the weights of
the model for us. Here we will use Adam; the optim package contains many other
optimizer which Variables it should update.
'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# creating batch loader for training
#trainloader = torch.utils.data.DataLoader(train_set, batch_size=N,
#                                          shuffle=True, num_workers=2)
train_iter = iter(trainloader)
#for t in range(500):
loss_val = 1000
t = 0
while loss_val> 1e-2:

    if t%int(58000/N)==0:
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=N,
                                                  shuffle=True, num_workers=2,
                                                  drop_last=True)
        # Drop_last drops the last imcomplete batch, if the dataset size is not
        # divisible by the batch size
        train_iter = iter(trainloader)
    # Making data into Variable anduforward prop
    images, labels = train_iter.next()
    # One hot encoding labels using sklearn preprocessign
    oneHot = preprocessing.OneHotEncoder()
    y = labels.numpy().reshape(N, 1)
    oneHot.fit(y)
    y = oneHot.transform(y).toarray()
    
    # Set up Variables and flatten x 
    x, y = Variable(images).view(N, 28, 28), Variable(torch.from_numpy(y), requires_grad=False)
    x = x.view(N, D_in)
    # Making sure the Variable type is FLoatTensor, since loss function 
    # only accepts FloatTensor
    x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    loss_val = loss.data[0]
    
    if t%20==0:
        print('using optimizer... t, loss: ', t, loss.data[0])


    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backprop
    loss.backward()

    # update the weights
    optimizer.step()
    
    t += 1





































