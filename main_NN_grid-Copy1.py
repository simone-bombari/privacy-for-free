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

# json_filename = f'{k}.json'
# if json_filename in os.listdir('./NN/trial_monday_22_07_airport'):
#     print(f"File {json_filename} already exists. Program will exit.")
#     sys.exit(0)

time.sleep(k)

save_dir = os.path.join('NN', '15_08_crema_grid_other_widths')  # to use together with airport, just more widths!
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
        total_grads[i] /= num_samples  # We average the gradients in third line of Alg 1

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

widths = [200, 5000, 10000]
network_width = widths[k % 3]
k = k // 3


model = SimpleFCN(network_width)

# lrs = [0.1]  # 0.2 seems better (maybe not onn large networks)
# lr = lrs[k % 2]
# k = k // 2  # 2 options

# Clr = 0.1
# lr = 1000 * Clr / network_width  # In RF lr depends on 1/p

Clr = 1  # Let's pump it!
lr = 1000 * Clr / network_width

n = 60000
delta = 1 / n

epsilons = [1, 2, 4]
eps = epsilons[k % 3]
k = k // 3  # 3 options

# clip_Cs = [20] 
# clip_C = clip_Cs[k % 2]
# k = k // 2  # 2 options
# clip_C = 20



'''
In 06_08_kohsamui_grid
Cs = np.logspace(0, 3, 20)
'''

Cs = np.logspace(-1, 3, 20)
# [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
clip_C = Cs[k % 20]  # 20 options
k = k // 20

clip_value = clip_C * np.sqrt(network_width / 1000)

'''
In 06_08_kohsamui_grid
Ts = np.logspace(1, 3.5, 20)
'''
# T = Ts[k % 2]
# k = k // 2  # 2 options

Ts = np.logspace(0, 3, 20)

# T = 200

for Tfloat in Ts:

    T = int(Tfloat)
    
    # tau = T * lr
    sigma = np.sqrt(lr * T) * 8 * np.sqrt(np.log(1 / delta)) / eps
    noise_magnitude = np.sqrt(lr) * (2 * clip_value / n) * sigma * np.sqrt(2)  # The last sqrt 2 is because I have two parameters.
    
    
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
    # metrics = {
    #     'n': n,
    #     'epsilon': eps,
    #     'clipC': clip_C,
    #     'T': T,
    #     'Clr': Clr,
    #     'network_width': network_width,
    #     'grad_norms': grad_norms,
    #     'train_losses': train_losses,
    #     'test_losses': test_losses,
    #     'test_accuracies': test_accuracies
    # }
    
    # save_path = os.path.join(save_dir, f'{args.k}.json')
    # with open(save_path, 'w') as f:
    #     json.dump(metrics, f)

    with open(os.path.join(save_dir, 'parameters_' + args.k + '.txt'), 'a') as f:
        f.write(
            f"{n}\t{network_width}\t{test_accuracy}\t{eps}\t{T}\t{clip_C:.2f}\n"
        )
    

