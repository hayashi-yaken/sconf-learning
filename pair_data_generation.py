import sys
import numpy as np
import scipy 
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from utils_algo import *
from models import *
class Dataset(object):
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)
        
    def __getitem__(self, index):
        return (self.x0[index], self.x1[index], self.label[index])

    def __len__(self):
        return self.size


def create_pairs(data, label, perm):
    x0_data = []
    x1_data = []
    label_sd = []
    for i in range(int(np.floor(len(label)/2))):
        x0_data.append(data[2*i])
        x1_data.append(data[2*i+1])
        label_sd.append(label[2*i])        
    x0_data = np.array(x0_data, dtype=np.float32)
    x1_data = np.array(x1_data, dtype=np.float32)
    label_sd = np.array(label_sd, dtype=np.int32)
    return x0_data, x1_data, label_sd


def create_iterator(data, label, perm):
    x0, x1, label_sd = create_pairs(data, label, perm)
    ret = Dataset(x0, x1, label_sd)
    return ret
