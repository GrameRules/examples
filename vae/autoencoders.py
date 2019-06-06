import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        self.__mu__, self.__logvar__ = self.encode(x.view(-1, 784))
        z = self.reparameterize(self.__mu__, self.__logvar__)
        return self.decode(z)

    def loss_function(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        a = torch.sum(1 + self.__logvar__ - self.__mu__.pow(2) - self.__logvar__.exp())
        KLD = -0.5 * torch.sum(1 + self.__logvar__ - self.__mu__.pow(2) - self.__logvar__.exp())

        return BCE + KLD

r"""
Also can be used as AE. just change input.
"""
class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x.view(-1, 784))
        x = self.decoder(x)
        return x

    def loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        return BCE

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x.view(-1, 784))
        x = self.decoder(x)
        return x

    def loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        return BCE