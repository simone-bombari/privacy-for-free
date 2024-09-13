import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import json
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--k')
args = parser.parse_args()

k = int(args.k)
time.sleep(k)


save_dir = os.path.join('NN', '03_09_grid_stack')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True  # Batch size of 1
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1024, shuffle=True, num_workers=1, pin_memory=True
)



class SimpleFCN(nn.Module):
    def __init__(self, width):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, width, bias=False)  # no bias
        self.fc2 = nn.Linear(width, 10, bias=False)  # no bias
        # nn.init.zeros_(self.fc2.weight)  # 0 init

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


        
def train_dp(model, train_loader, lr, n, criterion, device, clip_value, sigma):
    model.train()
    total_grads = [torch.zeros_like(param) for param in model.parameters()]

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # batch_size = 1
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))  # flatten and stack
        stacked_grads = torch.cat(grads)  # concatenate into a single vector

        grad_norm = torch.norm(stacked_grads)
        if grad_norm > clip_value:
            stacked_grads.mul_(clip_value / grad_norm)
        
        # split the stacked gradients back into the original parameter shapes
        start = 0
        for param in model.parameters():
            if param.grad is not None:
                end = start + param.grad.numel()
                param.grad.copy_(stacked_grads[start:end].view(param.grad.shape))
                start = end
        
        # accumulate gradients
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                total_grads[i] += param.grad

    for i in range(len(total_grads)):
        total_grads[i] /= n  # we average the gradients in the third line of Algorithm 1

    for i, param in enumerate(model.parameters()):
        noise = np.sqrt(lr) * (2 * clip_value / n) * sigma * torch.normal(0, 1, size=param.grad.size()).to(device)
        param.data = param.data - lr * total_grads[i] + noise



# def train_dp(model, train_loader, lr, n, criterion, device, clip_value, sigma):
#     model.train()
#     total_grads = [torch.zeros_like(param) for param in model.parameters()]

#     for data, target in train_loader:
#         data, target = data.to(device), target.to(device)  # Move data and target to device
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()

#         # Clip gradients layer-wise (parameter-wise)
        
#         for i, param in enumerate(model.parameters()):
#             if param.grad is not None:
#                 grad_norm = param.grad.norm()  # Compute the norm of the gradient for this layer
#                 if grad_norm > clip_value:
#                     param.grad.mul_(clip_value / grad_norm)  # Scale the gradient if it exceeds the clip_value
                
#                 # Accumulate gradients
#                 total_grads[i] += param.grad

#     # Average the accumulated gradients
#     for i in range(len(total_grads)):
#         total_grads[i] /= n  # Average the gradients over the number of samples

#     # Add noise and update the parameters
#     for i, param in enumerate(model.parameters()):
#         if param.grad is not None:
#             noise = np.sqrt(lr) * (2 * clip_value / n) * sigma * np.sqrt(2) * torch.normal(0, 1, size=param.grad.size()).to(device)
#             param.data = param.data - lr * total_grads[i] + noise



def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


widths = [100, 1000, 10000, 100000]
width = widths[k % 4]
k = k // 4

# Clr = 1
# lr = 1000 * Clr / width

# lrs = [0.1, 1]
# lr = lrs[k % 2]
# k = k // 2
lr = 0.1

n = 60000
delta = 1 / n

epsilons = [1, 4]
eps = epsilons[k % 2]
k = k // 2

# Cs = np.logspace(-1, 2, 10)
# clip_C = Cs[k % 10]  # 20 options
# k = k // 10

# clip_value = clip_C * np.sqrt(width)

clip_values = np.logspace(0, 2, 10)
clip_value = clip_values[k % 10]  # 10 options
k = k // 10

Tps = np.logspace(1, 3, 10)

for Tp in Tps:

    T = int(Tp * np.sqrt(10000 / width))
    print("------------------------------------")
    print(f"Training with T={T}")
    print("------------------------------------")
    
    model = SimpleFCN(width)

    sigma = np.sqrt(lr * T) * np.sqrt(8 * np.log(1 / delta)) / eps
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(1, T + 1):
        train_dp(model, train_loader, lr, n, criterion, device, clip_value, sigma)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%")
    
    with open(os.path.join(save_dir, 'parameters_' + args.k + '.txt'), 'a') as f:
        f.write(
            f"{n}\t{width}\t{test_accuracy}\t{eps}\t{T}\t{clip_value:.2f}\n"
        )
