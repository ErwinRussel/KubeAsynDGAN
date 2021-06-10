""" Definition of a KubeML function to train the LeNet network with the MNIST dataset"""
import logging
import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kubeml import KubeModel, KubeDataset
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    """ Discriminator
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        y = F.leaky_relu(self.fc1(x), 0.2)
        y = F.dropout(y, 0.3)
        y = F.leaky_relu(self.fc2(y), 0.2)
        y = F.dropout(y, 0.3)
        y = F.leaky_relu(self.fc3(y), 0.2)
        y = F.dropout(y, 0.3)
        y = torch.sigmoid(self.fc4(y))
        return y

class MnistDataset(KubeDataset):

    def __init__(self):
        super().__init__("mnist")

    def __getitem__(self, index):
        x_real = self.data[index]
        x_fake = self.labels[index]

        return x_real, x_fake

    def __len__(self):
        return len(self.data)


class KubeDiscriminator(KubeModel):

    def __init__(self, network: nn.Module, dataset: MnistDataset):
        super().__init__(network, dataset)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        adam = Adam(self.parameters(), lr=self.lr)
        return adam

    def init(self):
        pass

    def train(self, batch, batch_index) -> float:
        # define the device for training and load the data
        criterion = nn.BCELoss()
        total_loss = 0

        x_r, x_f = batch

        bs = batch[0].shape[0]

        self.optimizer.zero_grad()

        # train discriminator on real
        x_real, y_real = x_r.view(-1, 784), torch.ones(bs, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        output = self(x_real.reshape(bs,784))
        real_loss = criterion(output, y_real)

        # train discriminator on fake
        x_fake, y_fake = x_f, Variable(torch.zeros(bs, 1).to(device))

        output = self(x_fake.reshape(bs,784))
        fake_loss = criterion(output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        loss = real_loss + fake_loss
        loss.backward()

        # step with the optimizer
        self.optimizer.step()

        total_loss += loss.data.item()

        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")

        return total_loss

    def validate(self, batch, _) -> Tuple[float, float]:
        criterion = nn.BCELoss()

        x_r, x_f = batch

        bs = batch[0].shape[0]

        test_loss = 0
        correct = 0

        # test discriminator on real
        x_real, y_real = x_r.view(-1, 784), torch.ones(bs, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        output = self(x_real.reshape(bs, 784))
        real_loss = criterion(output, y_real)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y_real.view_as(pred)).sum().item()

        # test discriminator on fake
        x_fake, y_fake = x_f, Variable(torch.zeros(bs, 1).to(device))

        output = self(x_fake.reshape(bs, 784))
        fake_loss = criterion(output, y_fake)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y_fake.view_as(pred)).sum().item()

        loss = real_loss + fake_loss
        test_loss += loss.data.item()

        accuracy = 100. * correct / batch[0].shape[0]
        # self.logger.debug(f'accuracy {accuracy}')

        return accuracy, test_loss

    def infer(self, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass

def main():
    # set the random seeds
    torch.manual_seed(42)
    random.seed(42)

    discriminator = Discriminator()
    dataset = MnistDataset()
    kubedisc = KubeDiscriminator(discriminator, dataset)
    return kubedisc.start()

