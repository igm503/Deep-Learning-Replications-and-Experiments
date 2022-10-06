import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 500)  # 5*5 from image dimension
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.25)
        self.batch_norm_conv1 = nn.BatchNorm2d(6)
        self.batch_norm_conv2 = nn.BatchNorm2d(16)
        self.batch_norm1 = nn.BatchNorm1d(500)
        self.batch_norm2 = nn.BatchNorm1d(84)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_conv1(x)
        # If the size is a square, you can specify with a single number
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
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('test accuracy:', 100 * correct / total)

if __name__ == '__main__':

    # Load Data with additional 3x augmented data
    transform = transforms.ToTensor()
    augment = transforms.Compose([transforms.RandomRotation(10), transforms.RandomPerspective(distortion_scale=.2),transforms.ToTensor(), transforms.Normalize(0, 1)])
    batch_size = 500
    og_trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    augmented_trainset_1 = torchvision.datasets.MNIST(root='./data', train=True,
                                             transform=augment)
    augmented_trainset_2 = torchvision.datasets.MNIST(root='./data', train=True,
                                             transform=augment)
    augmented_trainset_3 = torchvision.datasets.MNIST(root='./data', train=True,
                                             transform=augment)
    trainset = torch.utils.data.ConcatDataset([augmented_trainset_1, augmented_trainset_2 , augmented_trainset_3, og_trainset])
    print(type(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=3, persistent_workers=True)
    # Load Eval Data
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1, persistent_workers=True)

    # Make Net and Loss Function
    device = 'mps'
    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    
    # Train Net
    for epoch in tqdm(range(20)): 
        net.train()
        running_loss = 0.0
        print('starting epoch')
        for i, data in tqdm(enumerate(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        print('epoch finished, running loss:', running_loss)
        net.eval()
        evaluate(testloader, net)