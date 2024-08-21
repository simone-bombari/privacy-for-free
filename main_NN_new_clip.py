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

json_filename = f'{k}.json'
if json_filename in os.listdir('./NN/trial_monday_22_07_new'):
    print(f"File {json_filename} already exists. Program will exit.")
    sys.exit(0)

time.sleep(k)

save_dir = os.path.join('NN', 'trial_monday_22_07_new')
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
        # Initialize the second layer with all zeros
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_dp(model, train_loader, optimizer, criterion, device, clip_value, clip_value_layer_1, noise_magnitude, grad_norms, train_losses):
    model.train()
    total_grads = [torch.zeros_like(param) for param in model.parameters()]
    batch_grad_norms = []
    epoch_loss = 0
    num_samples = 0
    clip_values = [clip_value_layer]

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # batch_size = 1
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        sample_grad_norms = []
        for param in model.parameters():
            if param.grad is not None:  # We have first and second layer
                grad_norm = param.grad.norm().item()
                sample_grad_norms.append(grad_norm)
        batch_grad_norms.append(sample_grad_norms)
        
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm > clip_value:
                    param.grad.mul_(clip_value / grad_norm)
                total_grads[i] += param.grad

        epoch_loss += loss.item()
        num_samples += 1

    print(num_samples)

    avg_grad_norms = np.mean(batch_grad_norms, axis=0).tolist()
    grad_norms.append(avg_grad_norms)
    
    avg_epoch_loss = epoch_loss / num_samples
    train_losses.append(avg_epoch_loss)

    for i in range(len(total_grads)):
        total_grads[i] /= num_samples  # We average the gradients in third line of Alg 1

    for i, param in enumerate(model.parameters()):
        noise_magnitudes = [noise_magnitude_layer_1, noise_magnitude]
        noise = noise_magnitudes[i] * torch.normal(0, 1, size=param.grad.size()).to(device)
        total_grads[i] += noise
        param.data -= optimizer.param_groups[0]['lr'] * total_grads[i]


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

widths = [250, 1000, 4000]
network_width = widths[k % 3]
k = k // 3  # 3 options

model = SimpleFCN(network_width)

lrs = [0.5, 0.2]  # 0.2 seems better
lr = lrs[k % 2]
k = k // 2  # 2 options

lr = 1000 * lr / network_width  # In RF lr depends on 1/p

n = 60000
delta = 1 / n

epsilons = [1, 10]
eps = epsilons[k % 3]
k = k // 3  # 3 options

clip_value_layer_1 = 5

clip_Cs = [5, 20] 
clip_C = clip_Cs[k % 2]
k = k // 2  # 2 options
clip_value = np.sqrt(network_width / 1000) * clip_C

Ts = [80, 160]
T = Ts[k % 2]
k = k // 2  # 2 options

# tau = T * lr
sigma = np.sqrt(lr * T) * 8 * np.sqrt(np.log(1 / delta)) / eps

noise_magnitude_layer_1 = np.sqrt(lr) * (2 * clip_value_layer_1 / n) * sigma
noise_magnitude = np.sqrt(lr) * (2 * clip_value / n) * sigma

print(network_width, lr)

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Lists to store gradients, losses, and accuracies
grad_norms = []
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(1, T + 1):
    print(f"Epoch {epoch}")
    train_dp(model, train_loader, optimizer, criterion, device, clip_value, clip_value_layer_1, noise_magnitude, noise_magnitude_layer_1, grad_norms, train_losses)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f"Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%")

# Save metrics to a dictionary and write to a file
metrics = {
    'epsilon': eps,
    'clipping_value': clip_value,
    'T': T,
    'lr': lr,
    'network_width': network_width,
    'grad_norms': grad_norms,
    'train_losses': train_losses,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies
}

save_path = os.path.join(save_dir, f'{args.k}.json')
with open(save_path, 'w') as f:
    json.dump(metrics, f)

print(f"Metrics saved to {save_path}")
