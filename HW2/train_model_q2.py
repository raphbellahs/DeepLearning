import torch
import torch.nn as nn
import glob
import unicodedata
import string
import random
import time
import math
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
all_letters = string.ascii_letters + " .,;'"


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    """
    unicodeToAscii("Καλημέρα") >>>'Καλημέρα'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    """
    This function reads the lines of a text file, normalizes the Unicode characters in each line
    using the 'unicodeToAscii' function, and returns a list of the preprocessed lines.
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def randomChoice(l):
    " Return a random item from a list"
    return l[random.randint(0, len(l) - 1)]


def randomTrainingPair():
    """ Get a random category and random line from that category """
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


def categoryTensor(category, all_categories):
    """
    This function creates a one-hot tensor for a given category.
    The tensor has a size of 1 x n_categories, where n_categories is the total number of categories.
    The element at the index of the input category is set to 1, and the other elements are set to 0.
    :param category: The category for which a one-hot tensor should be created.
    :return: The one-hot tensor for the input category.
    """
    li = all_categories.index(category)
    nbCategories = len(all_categories)
    tensor = torch.zeros(1, nbCategories)
    tensor[0][li] = 1
    return tensor


def inputTensor(line, n_letters):
    """
    Create a tensor from a given line of text.
    This function creates a tensor from a given line of text.
    The tensor has a size of len(line) x 1 x n_letters, where len(line) is the length of the input line and n_letters is the total number of letters in the alphabet.
    The element at the index of each letter in the input line is set to 1, and the other elements are set to 0.
    :param line: The line of text for which a tensor should be created.
    :return: The tensor for the input line.

    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def targetTensor(line):
    """
    LongTensor of second letter to end (EOS) for target
    :param line:
    :return:
    """
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample(all_category):
    """
    This function generates a random training example for the RNN model.
    The one-hot tensor for the category is generated using the 'categoryTensor' function.
    The input tensor for the line of text is generated using the 'inputTensor' function.
    The target tensor for the line of text is a tensor of the same size as the input tensor,
        with the same elements as the input tensor, except shifted one position to the right.
    :return: A tuple containing the category tensor, input tensor, and target tensor.
    """
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category, all_category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def train(model, category_tensor, input_line_tensor, target_line_tensor):
    """
    Train the RNN model on a single training example.
    This function trains the RNN model on a single training example.
    The final output and the average loss per element are returned.
    :param category_tensor: The category tensor.
    :param input_line_tensor:  The input tensor.
    :param target_line_tensor:  The target tensor.
    :return: A tuple containing the final output and the average loss per element.
    """
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()
    model.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


# Sample from a category and starting letter
def generateName(model, category, all_categories, n_letters, start_letter='A'):
    """
    Generate a name using the RNN model.
    :param category: The language for which a name should be generated.
    :param start_letter: The starting letter for the generated name. Default is 'A'.
    :return: The generated name.
    """
    max_length = 20
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, all_categories)
        input = inputTensor(start_letter, n_letters)
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = model(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter, n_letters)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(model, category, start_letters='ABC'):
    for start_letter in start_letters:
        print(generateName(model, category, start_letter))


if __name__ == '__main__':
    # Initialize the list of all letters in the English alphabet, as well as punctuation marks
    all_letters = string.ascii_letters + " .,;'"
    # Get the number of letters in the all_letters list
    n_letters = len(all_letters)

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in findFiles('data/names/*.txt'):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    # Get the number of categories (languages)
    n_categories = len(all_categories)

    # Initialize the Negative Log Likelihood Loss function as the criterion for training
    criterion = nn.NLLLoss()
    # Set the learning rate
    learning_rate = 0.001

    myRNN = RNN(n_letters, 128, n_letters, n_categories)
    # Set the number of iterations for training, as well as the frequency of printing and plotting losses
    n_iters = 100000
    print_every = 3000
    plot_every = 500
    # Initialize a list to store all the losses and a variable to keep track of the total loss
    all_losses = []
    total_loss = 0

    start = time.time()

    # TRAINING THE MODEL
    print('Training Started')
    for iter in range(1, n_iters + 1):
        category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample(all_categories)
        output, loss = train(myRNN, category_tensor, input_line_tensor, target_line_tensor)
        total_loss += loss
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
        if iter % plot_every == 0:
            # Add the average loss to the all_losses list and reset the total_loss variable
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    print('Training finished')
    # Plot the losses
    print('Plotting the losses')
    plt.plot(all_losses)
    plt.xlabel('500 Epochs')
    plt.ylabel('Loss')
    plt.legend(['Loss'])
    plt.title('Loss as a function of time (every step in the x-axis correspond to 500epochs)')
    plt.show()

    # Save the RNN model:
    torch.save(myRNN.state_dict(), 'myRNN.pkl')
