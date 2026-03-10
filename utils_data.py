import sys
import numpy as np
import scipy 
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from utils_algo import *
from models import *
from pair_data_generation import *
from scipy import stats

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def confidence_generator(train_loader, v_train_loader):
    model = mlp_model(input_dim=28*28,hidden_dim=500,output_dim=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    for epoch in range(10):
        for i, (images, label) in enumerate(train_loader):
            images, label = images.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = logistic(f=outputs, label=label)
            loss.backward()
            optimizer.step()
    conf = torch.Tensor([]).to(device)
    for i, (images, label) in enumerate(v_train_loader):
        images, label = images.to(device), label.to(device)
        a = model(images).detach().to(device)
        c = torch.sigmoid(a).to(device)
        conf = torch.cat((conf, c), 0).to(device)
    return conf

def prepare_mnist_data(batch_size):
    ordinary_train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset), shuffle=False)
    image = None
    labels = None
    for i, (image, labels) in enumerate(train_loader):
        continue
    for i in range(len(labels)):
        if labels[i]<7:
            labels[i] = 1
        else:
            labels[i] = -1
    train_dataset = torch.utils.data.TensorDataset(image, labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = False)


    v_train_loader = train_loader
    prior = torch.Tensor([0.7])
    
    conf = confidence_generator(train_loader, v_train_loader)
    perm = torch.randperm(len(labels))
    sconf = torch.zeros(len(labels)).to(device)
    for i in range(int(np.floor(len(labels)/2))):
        sconf[perm[2*i]] = conf[perm[2*i]]*conf[perm[2*i+1]] + (1 - conf[perm[2*i]])*(1 - conf[perm[2*i+1]])
        sconf[perm[2*i + 1]] = sconf[perm[2*i]]
    if len(labels)%2 == 1:
            sconf[perm[len(labels)-1]] = 0.5
    train_dataset = torch.utils.data.TensorDataset(image, sconf.to(device))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = True)

    sd_label = sconf.clone().detach().to(device)
    for i in range(int(np.floor(len(labels)/2))):
        sd_label[perm[2*i]] = torch.bernoulli(sd_label[perm[2*i]])
        sd_label[perm[2*i + 1]] = sd_label[perm[2*i]]        
    sd_train_dataset = torch.utils.data.TensorDataset(image, sd_label.to(device))
    sd_train_loader = torch.utils.data.DataLoader(dataset=sd_train_dataset, batch_size = batch_size, shuffle = True)
    
    pair_train_dataset = create_iterator(image.to("cpu").numpy(), sd_label.to("cpu").numpy(), perm)
    pair_train_loader = torch.utils.data.DataLoader(pair_train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    image = None
    labels = None
    for i, (image, labels) in enumerate(test_loader):
        continue
    for i in range(len(labels)):
        if labels[i]<7:
            labels[i] = 1
        else:
            labels[i] = -1
    test_dataset = torch.utils.data.TensorDataset(image, labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle = False)

    return v_train_loader, train_loader, test_loader, sd_train_loader, pair_train_loader, prior
