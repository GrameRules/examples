from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def add_noise(img):
    noise = torch.randn(img.size()) * 0.4
    noise = noise.to(device)
    noisy_img = img + noise
    return noisy_img

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


model = DAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)




def train(epoch):
    model.train()
    train_loss = 0
    train_loss_mle = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # add noise
        noisy_data = add_noise(data)
        noisy_data = noisy_data.to(device)
        optimizer.zero_grad()
        recon_batch = model(noisy_data)
        loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
        loss_mle = nn.MSELoss()(recon_batch, data.view(-1, 784), reduction='sum')
        loss.backward()
        train_loss += loss.item()
        train_loss_mle += loss_mle.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss BCE: {:.6f}, MLE: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                loss_mle.item() / len(data)
            ))

    print('====> Epoch: {} Training set loss bce: {:.4f}, mle: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), train_loss_mle / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_loss_mle = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # add noise
            noisy_data = add_noise(data)
            noisy_data = noisy_data.to(device)
            recon_batch = model(noisy_data)
            test_loss += F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
            test_loss_mle += nn.MSELoss()(recon_batch, data.view(-1, 784), reduction='sum')
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        noisy_data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/dae_rec_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    test_loss_mle /= len(test_loader.dataset)

    print('====> Test set loss bce: {:.4f}, mle: {:.4f}'.format(test_loss,test_loss_mle))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
