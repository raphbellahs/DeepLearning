import torch
import torchvision
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
from train_model_q1 import CNN

if __name__ == '__main__':

    # PREPARATION OF THE DATASET
    batch_size = 10000
    transform_test = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained model
    myCNN = CNN()
    myCNN.load_state_dict(torch.load('myCNN.pkl', map_location=lambda storage, loc: storage))

    # Check the accuracy on the test model
    myCNN.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = myCNN(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_acc = (100 * correct / total)
    
    print('Model successfully trained with test accuracy {}%'.format(test_acc))
