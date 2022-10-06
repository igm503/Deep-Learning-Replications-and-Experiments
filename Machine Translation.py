import torch 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.datasets import Multi30k

# Network Class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 5000) 
        self.fc2 = nn.Linear(5000, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.25)
        self.batch_norm_conv1 = nn.BatchNorm2d(6)
        self.batch_norm_conv2 = nn.BatchNorm2d(16)
        self.batch_norm1 = nn.BatchNorm1d(5000)
        self.batch_norm2 = nn.BatchNorm1d(84)

    def forward(self, x):
       pass
if __name__ == '__main__':
    # Load Data
    train, validation, test = Multi30k(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))
 

    # Create Loss Function, Optimizer, and Network

    # Train Network