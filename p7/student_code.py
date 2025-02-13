# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, bias=True)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=True)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_shape)
            x = self.maxpool1(self.relu1(self.conv1(dummy_input)))
            x = self.maxpool2(self.relu2(self.conv2(x)))
            flat_dim = x.numel()
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, 256, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes, bias=True)


    def forward(self, x):
        shape_dict = {}
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        shape_dict[1] = list(x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        shape_dict[2] = list(x.shape)
        x = self.flat(x)
        shape_dict[3] = list(x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        shape_dict[4] = list(x.shape)
        x = self.fc2(x)
        x = self.relu4(x)
        shape_dict[5] = list(x.shape)
        out = self.fc3(x)
        shape_dict[6] = list(out.shape)
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params += param.numel()
    model_params /= 1e6

    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
