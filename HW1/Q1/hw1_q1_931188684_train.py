#!/usr/bin/env python
# coding: utf-8

# # 2. Practical Part
# 
# ## 1. Create Neural Network from Scratch 
# 
# <h3 style='color:Blue'>
# Input Description
# </h3>
# 
# *Input* :
# The input consists of $m$ training images of shape 28x28 pixels. Consequently, each image is represented by a 1-dimensional array of size 784. In order to speed up the computations, I will take advantage of the vectorization technique. I will store the entire training set in a single matrix $X$. Each column of $X$ represents a training example:
# 
# $ X = 
#  \begin{bmatrix}
# \vdots & \vdots & \vdots & \vdots\\
# x^{(1)} & x^{(2)} &  ... & x^{(m)}\\
# \vdots & \vdots & \vdots & \vdots \\
# \end{bmatrix}
# $
# 
# 

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# The dimensions are : $X \in \mathbb{R}^{784xm} $

# In[18]:


# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


def getData():
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_dataset, train_loader,test_dataset,test_loader


##### Utils functions
def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def sigmoidPrime(s):
    # derivative of sigmoid
    # s: sigmoid output
    return s * (1 - s)


class Neural_Network:
    def __init__(self, input_size=784, output_size=10, hidden_size=32):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size
        self.loss = 0

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # self.W1 * X_train [10,000, 784]
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return sigmoid(self.z2)

    def backward(self, X, y, y_hat, lr=.1):
        # y holds the label with numbers form 0-9 [100]
        # t is the 1-hot version of y  [100,10]
        # y_hat looks like t

        t = torch.zeros(y_hat.size()).type(y.type())
        for n in range(t.size(0)):
            t[n][y[n]] = 1
        #         print(t.shape, y_hat.shape)

        batch_size = y.size(0)
        loss_batch = torch.sum((y_hat - t) ** 2) / batch_size

        self.loss = (loss_batch)

        dl_dz2 = (1 / batch_size) * (y_hat - t)
        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * sigmoidPrime(self.h)

        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def getLoss(self):
        return self.loss

    def saveModel(self):
        # Save the Model
        torch.save({'W1': self.W1,
                    'b1': self.b1,
                    'W2': self.W2,
                    'b2': self.b2}, 'model1.pkl')

    def loadModel(self, path):
        myModel = torch.load(path)
        self.W1 = myModel["W1"]
        self.b1 = myModel["b1"]
        self.W2 = myModel["W2"]
        self.b2 = myModel["b2"]

def getAccuracy(model,loader):
    correct,total = 0,0
    for i, (images, labels) in enumerate(loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
    return 100*correct / total

def plotAccuracy(trainAccuracy,testAccuracy):
    # Plot the train accuracy per epochs
    plt.plot(trainAccuracy, label='Train accuracy')
    plt.title('Train Accuracy per epochs on MNIST')
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot the train accuracy per epochs
    plt.plot(testAccuracy, label='Test accuracy')
    plt.title('Test Accuracy per epochs on MNIST')
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def TrainModel(model, train_loader,test_loader):
    # Train the model
    trainAccuracy = [0]
    testAccuracy = [0]

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            model.train(images, labels)

            if (i + 1) % 300 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, model.getLoss()))

        trainAccuracy.append(getAccuracy(model, train_loader))
        testAccuracy.append(getAccuracy(model, test_loader))

    # model.printWeight()


if __name__ == "__main__":
    train_dataset, train_loader,test_dataset,test_loader = getData()
    print('--- DATA FETCHED ---\n')

    model = Neural_Network()
    TrainModel(model, train_loader,test_loader)
    print('--- MODEL TRAINED ---\n')

    model.saveModel()
    print('--- MODEL SAVED ---\n')
