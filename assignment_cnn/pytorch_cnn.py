import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(in_features=28 * 28,
                            out_features=10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.fc = nn.Linear(in_features=28 * 28 * 10 // (2 * 2),
                            out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """
    def __init__(self):
        super(MyNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError("You have to implement this function.")
        return output


def classify(model, x):
    """
    :param model:    network model object
    :param x:        (batch_size, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_size, ) torch tensor with class labels
    """
    raise NotImplementedError("You have to implement this function.")
    return labels


if __name__ == '__main__':
    pass
