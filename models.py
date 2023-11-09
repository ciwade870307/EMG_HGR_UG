import numpy as np 
import matplotlib.pyplot as plt

# Torch
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Other
import argparse
import multiprocessing as mp
import os
import scipy.signal as signal

class DNN_feature(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False):
        super(DNN_feature, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            # nn.BatchNorm1d(12),
            nn.Linear(59, 128),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(64),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(32),
            # nn.Linear(32, 64),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, output_class)
        )

    def forward(self, x):
        return self.layers(x)
    
class CNN(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False):
        super(CNN, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,64, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),
            # nn.BatchNorm2d(32),
            nn.Conv2d(64,64, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),
            # nn.BatchNorm2d(16),
            nn.Conv2d(64,8 , kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(576, output_class)
            # nn.Linear(64, output_class)
        )
        self.conv = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding='same')
        self.maxpool = nn.MaxPool2d((4,1))

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        return self.layers(x)