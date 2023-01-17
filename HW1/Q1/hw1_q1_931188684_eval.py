import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from hw1_q1_931188684_train import Neural_Network

batch_size = 100

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getDataTest():
    # MNIST dataset
    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return test_dataset, test_loader


def testModel(myTrainedModel, test_loader):
    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = myTrainedModel.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == "__main__":
    test_dataset, test_loader = getDataTest()
    print('--- DATA FETCHED ---\n')

    myTrainedModel = Neural_Network()
    myTrainedModel.loadModel('model.pkl')

    testModel(myTrainedModel, test_loader)
