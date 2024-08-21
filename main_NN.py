import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--k')
args = parser.parse_args()

k = int(args.k)

time.sleep(k)

save_dir = os.path.join('NN', 'trial_fri_0')
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
        self.fc1 = nn.Linear(28 * 28, width, bias=False)  # No bias
        self.fc2 = nn.Linear(width, 10, bias=False)  # No bias

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_dp(model, train_loader, optimizer, criterion, device, clip_value, noise_scale):
    model.train()
    train_losses = []
    total_grads = [torch.zeros_like(param) for param in model.parameters()]
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Clip and aggregate gradients
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                # print(param.shape)
                grad_norm = param.grad.norm()
                if grad_norm > clip_value:
                    param.grad.mul_(clip_value / grad_norm)
                total_grads[i] += param.grad

    # print('parameter ', i, 'param_shape ', param.shape, 'grad_norm ', grad_norm, flush=True)
    for i, param in enumerate(model.parameters()):
        noise = torch.normal(0, noise_scale, size=param.grad.size()).to(device)
        total_grads[i] += noise
        param.data -= optimizer.param_groups[0]['lr'] * total_grads[i]
    
    print(f"Train Loss: {loss.item():.6f}")




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
    
    print(f"Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.2f}%")



widths = [128, 256, 512, 1024, 2048, 4096]
network_width = widths[k % 6]  # 6 options
k = k // 6

model = SimpleFCN(network_width)

lrs = [1e-5, 5e-6, 2e-6, 1e-6]
lr = lrs[k % 4]
k = k // 4    # 4 options

print(network_width, lr)

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

clip_value = 1000
noise_scale = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

for epoch in range(1, 101):
    print(f"Epoch {epoch}")
    train_dp(model, train_loader, optimizer, criterion, device, clip_value, noise_scale)
    test(model, test_loader, criterion, device)


