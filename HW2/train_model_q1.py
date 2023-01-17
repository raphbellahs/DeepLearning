import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt

import os
import warnings

warnings.filterwarnings("ignore")


def getScore(loader, model):
    # Calculate accuracy on training set
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
        error = 1 - acc
        return acc, error


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(4 * 4 * 64, 16)
        self.fc2 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.logsoftmax(out)
        out = self.fc2(out)
        return self.logsoftmax(out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Hyper-parameters
    num_epochs = 90
    batch_size = 128
    learning_rate = 0.001

    # Image Preprocessing and data augmentation
    transform_train = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.RandomHorizontalFlip(p=0.5),  # FLips the image w.r.t horizontal axis
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
         ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR-10 Dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)

    # Define the model, criterion, optimizer, and number of epochs
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    # model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))

    # Lists to store losses and accuracies
    train_losses = []
    test_losses = []
    train_errs = []
    test_errs = []

    for epoch in range(num_epochs):
        # Train data
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = cnn(images)
            train_loss = criterion(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (i + 1) % 250 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, i + 1,
                                                                  len(train_dataset) // batch_size, train_loss.data))

        # Save the train loss for the epoch
        train_losses.append(train_loss.item())

        # Test data
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                # Forward Only
                outputs = cnn(images)
                test_loss = criterion(outputs, labels)
            # Save the train loss for the epoch
            test_losses.append(test_loss.item())

        # Calculate accuracy on training set
        train_acc, train_error = getScore(train_loader, cnn)
        train_errs.append(train_error)

        # Calculate accuracy on test set
        test_acc, test_error = getScore(test_loader, cnn)
        test_errs.append(test_error)

    # Plot the losses and accuracies
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    plt.title('Convergence graph: loss as a function of time (epochs)')
    plt.show()

    # Plot the losses and accuracies
    plt.plot(train_errs)
    plt.plot(test_errs)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(['Train', 'Test'])
    plt.title('Convergence graph: error as a function of time (epochs)')
    plt.show()

    cnn.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_acc = (100 * correct / total)

    print(f"The final accuracy is {test_acc}")

    # Step 8:
    # Save the Trained Model
    torch.save(cnn.state_dict(), 'myCNN.pkl')
