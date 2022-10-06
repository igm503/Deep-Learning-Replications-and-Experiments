import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class digit_reader(nn.Module):
    def __init__(self, model_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6 * model_size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6 * model_size, 16 * model_size, 5)
        self.fc1 = nn.Linear(16 * model_size * 4 * 4, 500 * model_size)  # 5*5 from image dimension
        self.fc2 = nn.Linear(500 * model_size, 84 * model_size)
        self.fc3 = nn.Linear(84 * model_size, 10)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm_conv1 = nn.BatchNorm2d(6 * model_size)
        self.batch_norm_conv2 = nn.BatchNorm2d(16 * model_size)
        self.batch_norm1 = nn.BatchNorm1d(500 * model_size)
        self.batch_norm2 = nn.BatchNorm1d(84 * model_size)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_conv1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_conv2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.fc3(x)
        return x

def evaluate(testloader, net):
    with torch.no_grad():
        running_loss = 0
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss

if __name__ == '__main__':
    transform = transforms.ToTensor()
    batch_size_list = [64, 256, 512, 1024, 4096]
    data_size_list = [1, .5, .25, .125, .0625]
    model_size_list = [1, 2, 4, 8, 16]
    device = 'mps'

    results = np.ones((5, 5, 5))

    for batch_size in tqdm(batch_size_list):
         # Load Training Data
        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
        # Load Eval Data
        test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=1, persistent_workers=True)
        # Load to device
        train_data.data.to(device)  
        train_data.targets.to(device)
        
        for model_size in model_size_list:
            for data_size in data_size_list:
                # Make Net and Loss Function
                net = digit_reader(model_size)
                net = net.to(device)
                criterion = nn.CrossEntropyLoss()
                rate = 0.01 / np.sqrt(model_size)
                optimizer = optim.Adam(net.parameters(), lr=rate, betas=(0.9, 0.98), eps=1e-9)
                print(f'model size: {model_size}, data size: {data_size}, batch size: {batch_size}')
                for i, data in tqdm(enumerate(train_loader)):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(device), data[1].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad(set_to_none=True)
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    if i > (data_size * 50000) / batch_size:
                        break
                net.eval()
                eval_loss = evaluate(test_loader, net)
                x, y, z = batch_size_list.index(batch_size), model_size_list.index(model_size), data_size_list.index(data_size)
                results[x, y, z] = eval_loss
                print('test loss:', eval_loss)

if __name__ == '__main__':
 
    accuracy_list16 = []
    accuracy_list64 = []
    accuracy_list256 = []
    accuracy_list512 = []
    accuracy_list1024 = []
    compute_list = []
    for y in range(5):
        for z in range(5):
            accuracy_list16.append(results[0, y, z])
            accuracy_list64.append(results[1, y, z])
            accuracy_list256.append(results[2, y, z])
            accuracy_list512.append(results[3, y, z])
            accuracy_list1024.append(results[4, y, z])
            compute_list.append(4**y / 2**z)

    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1)
    ax1.scatter(compute_list, accuracy_list16)
    ax1.set_xscale('log')``
    ax2.scatter(compute_list, accuracy_list16, label = 'batchsize: 16')
    ax2.scatter(compute_list, accuracy_list64, label = 'batchsize: 64')
    ax2.scatter(compute_list, accuracy_list256, label = 'batchsize: 256')
    ax2.scatter(compute_list, accuracy_list512, label = 'batchsize: 512')
    ax2.scatter(compute_list, accuracy_list1024, label = 'batchsize: 1024')
    ax2.set_xscale('log')
    plt.show()
'''
Training for a single epoch only, vary the model size and dataset size. 
For the model size, multiply the width by powers of sqrt(2) 
(rounding if necessary - the idea is to vary the amount of compute used per forward pass by powers of 2). 

For the dataset size, multiply the fraction of the full dataset used by powers of 2 (i.e. 1, 1/2, 1/4, ...). 

To reduce noise, use a few random seeds and always use the full validation set.

The learning rate will need to vary with model size. Either tune it carefully for each model size, 
or use the rule of thumb that for Adam, the learning rate should be proportional to the initialization scale, 
i.e. 1/sqrt(fan_in) for the standard Kaiming He initialization (which is what PyTorch generally uses by default).

Plot the amount of compute used (on a log scale) against validation loss. The compute-efficient frontier should 
follow an approximate power law (straight line on a log scale).

How does validation accuracy behave?

Study how the compute-efficient model size varies with compute. This should also follow an approximate power law. 
Try to estimate its exponent.

Repeat your entire experiment with 20% dropout to see how this affects the scaling exponents.'''
   