"""
Adversarially train GoogLeNet
"""

import torch
import torch.nn as nn
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt

import os

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test

#from models import LeNet5


# Hyper-parameters
param = {
    'batch_size': 96,
    'test_batch_size': 16,
    'num_epochs': 5,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
}


# Data loadersvtype_train = datasets.ImageFolder(os.path.join("/usr/home/st119220/torchex1/data/train"), transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
vtype_train = datasets.ImageFolder(os.path.join("/usr/home/st119220/torchex1/data/train"), transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
vtype_test = datasets.ImageFolder(os.path.join("/usr/home/st119220/torchex1/data/val"), transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
loader_train = DataLoader(vtype_train, batch_size = param['batch_size'], shuffle=True)
loader_test = DataLoader(vtype_test, batch_size = param['test_batch_size'], shuffle=True)


# Setup the model
net = models.googlenet(aux_logits=False, num_classes=5)

#if torch.cuda.is_available():
#    print('CUDA ensabled.')
#    net.cuda()
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
net.train()

# Adversarial training setup
adversary = FGSMAttack(net, epsilon=0.3)
#adversary = LinfPGDAttack()

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
    weight_decay=param['weight_decay'])

for epoch in range(param['num_epochs']):

    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

    for t, (x, y) in enumerate(loader_train):

        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)

        # adversarial training
        if epoch+1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), y_var)
            loss = (loss + loss_adv) / 2

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test(net, loader_test)

torch.save(net.state_dict(), 'models/adv_trained_googlenet.pth')
