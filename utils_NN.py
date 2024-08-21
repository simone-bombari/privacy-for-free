import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
import os
import numpy as np


def load_full_dataset(dataset, N, Nt):
    
    if 'MNIST' in dataset:
        dataset = 'MNIST'
    elif 'CIFAR-10' in dataset:
        dataset = 'CIFAR-10'
    
    train_dir = './data/' + os.path.join(dataset, 'train')
    test_dir = './data/' + os.path.join(dataset, 'test')

    data_train = np.load(train_dir + '/augmented_data.npy')
    labels_train = np.load(train_dir + '/labels.npy')
    data_test = np.load(test_dir + '/augmented_data.npy')
    labels_test = np.load(test_dir + '/labels.npy')
    
    if N > augmented_data_train.shape[0]:
        N = augmented_data_train.shape[0]
    if Nt > augmented_data_test.shape[0]:
        Nt = augmented_data_test.shape[0]
    
    indices = np.random.choice(range(augmented_data_train.shape[0]), size=N, replace=False, p=None)
    indices_t = np.random.choice(range(augmented_data_test.shape[0]), size=Nt, replace=False, p=None)

    return augmented_data_train[indices], augmented_data_test[indices_t], attack_data[indices], labels_train[indices], labels_test[indices_t]


class FullyConnected(nn.Module):
    def __init__(self, dataset='MNIST', noise_width=0):
        
        super(FullyConnected, self).__init__()
        if dataset == 'synthetic':
            d = noise_width
            num_classes = 1
        else:
            num_classes = 10 if dataset == 'CIFAR-10' else 100 if dataset == 'CIFAR-100' else 10
            in_channels = 1 if dataset == 'MNIST' else 3 if dataset == 'CIFAR-10' else 3
            img_size = 28 + 2 * noise_width if dataset == 'MNIST' else 32 + 2 * noise_width if dataset == 'CIFAR-10' else 32
            d = in_channels * img_size ** 2
        
        self.fc1 = nn.Linear(d, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SmallConvNet(nn.Module):
    def __init__(self, dataset='MNIST', noise_width=0):
        super(SmallConvNet, self).__init__()
        self.dataset = dataset
        num_classes = 10 if dataset == 'CIFAR-10' else 100 if dataset == 'CIFAR-100' else 10
        in_channels = 1 if dataset == 'MNIST' else 3 if dataset == 'CIFAR-10' else 3
        img_size = 28 + 2 * noise_width if dataset == 'MNIST' else 32 + 2 * noise_width if dataset == 'CIFAR-10' else 32        
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * (img_size // 4) * (img_size // 4), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        if self.dataset == 'MNIST':
            x = torch.unsqueeze(x, dim=1)
        elif self.dataset == 'CIFAR-10':    
            x = x.permute(0, 3, 1, 2)
            
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def loss_calculator(outputs, labels, loss_function):
    if loss_function == 'MSE':
        criterion = nn.MSELoss()
        labels = labels.unsqueeze(1)
    elif loss_function == 'CEL':
        criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


def compute_loss_accuracy_regressor(data_loader, loss_function, net, device):
    score = 0
    samples = 0
    full_loss = 0

    for input_images, labels in iter(data_loader):
        input_images, labels = input_images.to(device), labels.to(device)
        outputs = net(input_images)
        minibatch_loss = loss_calculator(outputs, labels, loss_function).item()
        predicted = torch.sign(outputs.squeeze())
        labels = torch.sign(labels)  # To give an accuracy on this regression task

        minibatch_score = (predicted == labels).sum().item()
        minibatch_size = len(labels)
        score += minibatch_score
        full_loss += minibatch_loss * minibatch_size
        samples += minibatch_size

    loss = full_loss / samples
    accuracy = score / samples

    return loss, accuracy


def compute_loss_accuracy_classifier(data_loader, loss_function, net, device):
    score = 0
    samples = 0
    full_loss = 0

    for input_images, labels in iter(data_loader):
        input_images, labels = input_images.to(device), labels.to(device)
        outputs = net(input_images)
        minibatch_loss = loss_calculator(outputs, labels, loss_function).item()
        predicted = torch.max(outputs, 1)[1]

        minibatch_score = (predicted == labels).sum().item()
        minibatch_size = len(labels)
        score += minibatch_score
        full_loss += minibatch_loss * minibatch_size
        samples += minibatch_size

    loss = full_loss / samples
    accuracy = score / samples

    return loss, accuracy


class BigConvNet(nn.Module):
    def __init__(self, dataset='MNIST', noise_width=0):
        
        super(BigConvNet, self).__init__()
        self.dataset = dataset
        
        num_classes = 10 if dataset == 'CIFAR-10' else 100 if dataset == 'CIFAR-100' else 10
        in_channels = 1 if dataset == 'MNIST' else 3 if dataset == 'CIFAR-10' else 3
        img_size = 28 + 2 * noise_width if dataset == 'MNIST' else 32 + 2 * noise_width if dataset == 'CIFAR-10' else 32
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([32, img_size, img_size]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, img_size // 2, img_size // 2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([128, img_size // 4, img_size // 4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(128 * (img_size // 8) ** 2, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        if self.dataset == 'MNIST':
            x = torch.unsqueeze(x, dim=1)
        elif self.dataset == 'CIFAR-10':    
            x = x.permute(0, 3, 1, 2)
            
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
