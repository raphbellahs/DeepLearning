# Import necessary libraries
import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from hw1_q2_931188684_train import NeuralNetwork, generateRandomLabel, get_data

if __name__ == "__main__":
    # Get MNIST Dataset (Images and Labels)
    train_loader, test_loader = get_data(batch_size=128)

    # Load saved torch model
    model = NeuralNetwork()
    model.load_state_dict(torch.load('modelRandom.pkl'))
    model.eval()

    random_labels = generateRandomLabel(128)

    # Variables to store our loss per epoch
    lossTrain = []
    lossTest = []

    # Run inference on the train model
    for first_batch in train_loader:
        images, labels = first_batch
        break

    train_output = model(images.view(-1, 28 * 28))
    lossTrain = F.cross_entropy(train_output, random_labels)

    # Run inference on the test model
    for first_batch in test_loader:
        images, labels = first_batch
        break
    test_output = model(images.view(-1, 28 * 28))
    lossTest = F.cross_entropy(test_output, random_labels)

    print("The loss on the Train : " + str(lossTrain.item()))
    print("The loss on the Test  : " + str(lossTest.item()))
