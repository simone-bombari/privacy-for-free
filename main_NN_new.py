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
from torch.utils.data import DataLoader, Subset


parser = argparse.ArgumentParser()
parser.add_argument('--k')
args = parser.parse_args()

k = int(args.k)

time.sleep(k)

save_dir = os.path.join('NN', '08_14_crema')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_random_subset_sampler(dataset, n):
    indices = np.random.choice(len(dataset), n, replace=False)
    return torch.utils.data.SubsetRandomSampler(indices)

# ns = [600, 6000, 60000]
# n = ns[k % 3]
# k = k // 3  # 3 options

n = 60000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
subset_sampler = get_random_subset_sampler(full_train_dataset, n)
train_loader = DataLoader(full_train_dataset, batch_size=1, sampler=subset_sampler, num_workers=1, pin_memory=True)


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=1, shuffle=True, num_workers=1, pin_memory=True  # Batch size of 1
# )

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


def train_dp(model, train_loader, optimizer, criterion, device, clip_value, noise_magnitude, grad_norms, train_losses):
    model.train()
    total_grads = [torch.zeros_like(param) for param in model.parameters()]
    batch_grad_norms = []
    epoch_loss = 0
    num_samples = 0

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
        total_grads[i] /= num_samples  # We average the gradients

    for i, param in enumerate(model.parameters()):
        noise = noise_magnitude * torch.normal(0, 1, size=param.grad.size()).to(device)
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

widths = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
network_width = widths[k % 10]
k = k // 10  # 10 options

model = SimpleFCN(network_width)

# lrs = [0.1]  # 0.2 seems better (maybe not onn large networks)
# lr = lrs[k % 2]
# k = k // 2  # 2 options

Clr = 0.1
lr = 1000 * Clr / network_width  # In RF lr depends on 1/p

# n = 60000
delta = 1 / n

epsilons = [0.01, 1, 2, 4, 100]
eps = epsilons[k % 5]
k = k // 5  # 5 options

# clip_Cs = [20] 
# clip_C = clip_Cs[k % 2]
# k = k // 2  # 2 options
clip_C = 25

clip_value = clip_C * np.sqrt(network_width / 1000)

# Ts = [200]
# T = Ts[k % 2]
# k = k // 2  # 2 options

T = 400

# tau = T * lr
sigma = np.sqrt(lr * T) * 8 * np.sqrt(np.log(1 / delta)) / eps
noise_magnitude = np.sqrt(lr) * (2 * clip_value / n) * sigma * np.sqrt(2)  # The last sqrt 2 is because I have two parameters.

# print(network_width, lr)

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
    train_dp(model, train_loader, optimizer, criterion, device, clip_value, noise_magnitude, grad_norms, train_losses)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f"Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%")

# Save metrics to a dictionary and write to a file
metrics = {
    'n': n,
    'epsilon': eps,
    'clipC': clip_C,
    'T': T,
    'Clr': Clr,
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
