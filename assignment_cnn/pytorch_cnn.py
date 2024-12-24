import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
         
    def forward(self, x):
        output = self.layer(x)
        return output
 
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.ReLU())
    def forward(self, x):
        return self.layer(x)

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
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
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16)
        )

        self.classifier = nn.Sequential(
            MLPBlock(7*7*16, 256),
            nn.Linear(256, 10)
        ) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


class FinetuneNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """
    def __init__(self, load_pretrained=False):
        super().__init__()
        
        if load_pretrained:
            # only during training, not in BRUTE
            self.model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            # used during evaluation in BRUTE - no weights downloads here!
            self.model = torchvision.models.vit_b_16()
            
        self.model.heads = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 10)
        ) 

        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        resized_x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.model(resized_x)


def classify(model, x):
    """
    :param model:    network model object
    :param x:        (batch_size, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_size, ) torch tensor with class labels
    """

    logits = model(x)
    labels = torch.argmax(logits, dim=1)

    return labels


if __name__ == '__main__':
    pass
