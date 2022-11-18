import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, model_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, int(2.3 * model_size), 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(int(2.3 * model_size), int(3 * model_size), 5)
        self.fc1 = nn.Linear(int(3 * model_size) * 4 * 4, int(30 * model_size)) 
        self.fc2 = nn.Linear(int(30 * model_size), int(10 * model_size))
        self.fc3 = nn.Linear(int(10 * model_size), 10)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm_conv1 = nn.BatchNorm2d(int(2.3 * model_size))
        self.batch_norm_conv2 = nn.BatchNorm2d(int(3 * model_size))
        self.batch_norm1 = nn.BatchNorm1d(int(30 * model_size))
        self.batch_norm2 = nn.BatchNorm1d(int(10 * model_size))
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_conv1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_conv2(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.fc3(x)
        return x