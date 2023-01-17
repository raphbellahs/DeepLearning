import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_data(batch_size=128):
    """
    This function downloads the MNIST dataset and returns the train and test data loaders.
    """
    train = datasets.MNIST("", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=200):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x, dim=1)


def train(iterations, net, random_labels):
    """
        This function takes in a neural network, a dataset, and a number of iterations, and returns the neural network
        after training it on the dataset for the given number of iterations.
        Parameters:
    """

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Work only on the first 128 samples from the training dataset.
    for first_batch in train_loader:
        images, labels = first_batch
        break

    if iterations == 0:
        # Compute the cross-entropy loss
        loss = F.cross_entropy(images.view(-1, 784), random_labels)
    else:
        for epoch in range(iterations):
            net.zero_grad()
            output = net(images.view(-1, 784))
            loss = F.cross_entropy(output, random_labels)
            loss.backward()
            optimizer.step()

    return net, loss


def plotGraph(epochList, lossTrain, lossTest):
    # Plot the train accuracy per epochs
    plt.plot(lossTrain, label='Train loss.')
    plt.title('Train loss per epochs on MNIST')
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot the train accuracy per epochs
    plt.plot(lossTest, label='Test loss.')
    plt.title('Test loss per epochs on MNIST')
    plt.ylabel('Accuracy in %')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


def generateRandomLabel(sizeVector):
    # Generate random labels from Bernoulli distribution with a probability of 1⁄2.
    torch.manual_seed(12)
    vec = torch.empty(1, sizeVector).uniform_(0, 1)
    vec = torch.bernoulli(vec)
    vec = vec[0].type(torch.LongTensor)
    return vec


if __name__ == "__main__":
    # Get MNIST Dataset (Images and Labels)
    train_loader, test_loader = get_data(batch_size=128)
    print('--- DATA FETCHED ---\n')

    # Variables to store our loss per epoch
    lossTrain = []
    lossTest = []

    random_labels = generateRandomLabel(128)

    # Run
    epochList = [i for i in range(301)]
    for i in epochList:
        if i % 100 == 0:
            print('Epoch: ' + str(i))

        net = NeuralNetwork()
        iteration = i
        trainedNet, train_loss = train(iteration, net, random_labels)
        lossTrain.append(train_loss.item())

        for first_batch in test_loader:
            images, labels = first_batch
            break
        output = net(images.view(-1, 28 * 28))
        test_lost = F.cross_entropy(output, random_labels)
        lossTest.append(test_lost.item())

    print("The loss on the Train data is : " + str(lossTrain[300]))
    print("The loss on the Test data is : " + str(lossTest[300]))

    torch.save(net.state_dict(), 'modelRandom.pkl')


    plotGraph(epochList, lossTrain, lossTest)

