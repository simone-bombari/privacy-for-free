# Step 1: Importing Libraries
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Step 2: Loading MNIST Data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True  # Batch size of 1
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1024, shuffle=True, num_workers=1, pin_memory=True
)

# Step 3: Defining a Simple 2-layer Fully Connected Neural Network
class SimpleFCN(nn.Module):
    def __init__(self, width):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, width)
        self.fc2 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the width of the network
network_width = 128  # Customize the width here

model = SimpleFCN(network_width)

# Step 4: Define the Optimizer and Loss Function
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

# Clipping and noise constants (set these values as needed)
clip_value = 1.0
noise_scale = 0.1

# Step 5: Training the Model with DP-GD
def train_dp(model, train_loader, optimizer, criterion, device, clip_value, noise_scale):
    model.train()
    train_losses = []
    total_grads = [torch.zeros_like(param) for param in model.parameters()]
    
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Clip and aggregate gradients
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm > clip_value:
                    param.grad.mul_(clip_value / grad_norm)
                total_grads[i] += param.grad
    
    # Add noise and update parameters
    for i, param in enumerate(model.parameters()):
        noise = torch.normal(0, noise_scale, size=param.grad.size()).to(device)
        total_grads[i] += noise
        param.data -= optimizer.param_groups[0]['lr'] * total_grads[i]
    
    print(f"Train Loss: {loss.item():.6f}")

# Step 6: Evaluating the Model
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

# Main Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 11):
    print(f"Epoch {epoch}")
    train_dp(model, train_loader, optimizer, criterion, device, clip_value, noise_scale)
    test(model, test_loader, criterion, device)
