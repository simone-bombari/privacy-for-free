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

save_dir = os.path.join('NN', '03_09_GD_0init')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=60000, shuffle=True, num_workers=1, pin_memory=True  # Full batch
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=10000, shuffle=True, num_workers=1, pin_memory=True
)


class SimpleFCN(nn.Module):
    def __init__(self, width):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, width, bias=False)  # no bias
        self.fc2 = nn.Linear(width, 10, bias=False)  # no bias
        nn.init.zeros_(self.fc2.weight)  # 0 init

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def train_full_batch(model, train_loader, optimizer, criterion, device, sample_size=100):
    model.train()
    param1_grad_norms = []
    param2_grad_norms = []

    # Get the full batch of data
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    # Calculate the gradient norms for a subset of the samples
    for i in range(sample_size):
        sample_grad_norm_param1 = 0
        sample_grad_norm_param2 = 0
        model.zero_grad()  # Reset gradients for each sample
        output = model(data[i].unsqueeze(0))  # Forward pass for the single sample
        loss = criterion(output, target[i].unsqueeze(0))  # Compute loss for the sample
        loss.backward()  # Compute gradients for this sample
    
        for j, param in enumerate(model.parameters()):
            if param.grad is not None:
                if j == 0:  # First parameter (hidden layer)
                    sample_grad_norm_param1 += param.grad.norm().item() ** 2
                elif j == 1:  # Second parameter (output layer)
                    sample_grad_norm_param2 += param.grad.norm().item() ** 2

        param1_grad_norms.append(np.sqrt(sample_grad_norm_param1))
        param2_grad_norms.append(np.sqrt(sample_grad_norm_param2))

    # After computing the per-sample gradients, perform the actual full-batch optimization step
    optimizer.zero_grad()  # Reset gradients before full batch forward pass
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Compute gradients for the full batch

    optimizer.step()  # Perform the optimizer step after computing gradients for the full batch

    avg_grad_norm_param1 = np.mean(param1_grad_norms)
    avg_grad_norm_param2 = np.mean(param2_grad_norms)

    return loss.item(), avg_grad_norm_param1, avg_grad_norm_param2




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

    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


widths = [100, 200, 500, 1000, 2000, 5000, 10000]
width = widths[k % 7]
k = k // 7

model = SimpleFCN(width)

lrs = [0.0001, 0.001, 0.01, 0.1, 1] # , 0.1, 1, 10]
lr = lrs[k % 5]
k = k // 5
# lr = 1000 * Clr / width

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
test_losses = []
test_accuracies = []
grad_norms_param1 = []
grad_norms_param2 = []

T = 500

for epoch in range(1, T + 1):
    print(f"Epoch {epoch}")
    train_loss, avg_grad_norm_param1, avg_grad_norm_param2 = train_full_batch(model, train_loader, optimizer, criterion, device)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    grad_norms_param1.append(avg_grad_norm_param1)
    grad_norms_param2.append(avg_grad_norm_param2)
    
    print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%")
    print(f"Avg Grad Norm Param1: {avg_grad_norm_param1:.6f}, Avg Grad Norm Param2: {avg_grad_norm_param2:.6f}")

# Save results to JSON file
results = {
    'width': width,
    'lr': lr,
    'train_losses': train_losses,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'grad_norms_param1': grad_norms_param1,
    'grad_norms_param2': grad_norms_param2
}

with open(os.path.join(save_dir, f'results_{args.k}.json'), 'w') as f:
    json.dump(results, f)
