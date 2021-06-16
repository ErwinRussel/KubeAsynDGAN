import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json 
import subprocess, os
import re
import threading
from redisai import Client
import time
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# todo kill this process
    
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

class TrainGenerator:
    def __init__(self, job_id):
        # Load latest generator model
        self.PATH = "models/latest_model"
        self.G = Generator(g_input_dim = 100, g_output_dim = 784).to(device)
        self.G.load_state_dict(torch.load(self.PATH))
        self.G.eval()

        self.criterion = nn.BCELoss()
        self.lr = 0.0002
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = self.lr)

        self.G_losses = []
        self.dataset_size = 10000
        self.batch_size = 64

        self.D_latest = Discriminator()

        con = Client(host='localhost', port=6379)

        print("- Getting latest model")
        state_dict = dict()
        for name in self.D_latest.state_dict():
            # load each of the layers in the statedict
            weight_key = f'{job_id}:{name}'
            w = con.tensorget(weight_key)
            # set the weight
            state_dict[weight_key[len(job_id) + 1:]] = torch.from_numpy(w)

        self.D_latest.load_state_dict(state_dict)

        self.n_epochs = 100

    def train(self):
        G_losses = []
        # TODO remove this loader stuff
        for i in range(int(self.dataset_size/self.batch_size)):
            z = Variable(torch.randn(self.batch_size, 100).to(device))
            y = Variable(torch.ones(self.batch_size, 1).to(device))
            G_output = self.G(z)

            D_output = self.D_latest(G_output)

            G_loss = self.criterion(D_output, y)

            # # gradient backprop & optimize ONLY G's parameters
            G_loss.backward()
            self.G_optimizer.step()
            G_losses.append(G_loss.data.item())

        print("- Loss: " + str(torch.mean(torch.FloatTensor(G_losses))))
        print("- Saving new model")
        torch.save(self.G.state_dict(), self.PATH)

