from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from ae_utils import add_noise

def train(epoch,data_loader,model,device,args):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss = 0
    train_loss_mle = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        # add noise
        noisy_data = add_noise(data,device)
        noisy_data = noisy_data.to(device)
        optimizer.zero_grad()
        recon_batch = model(noisy_data)
        loss = model.loss(recon_batch, data.view(-1, 784))
        loss_mle = nn.MSELoss(reduction='sum')(recon_batch, data.view(-1, 784))
        loss.backward()
        train_loss += loss.item()
        train_loss_mle += loss_mle.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss BCE: {:.6f}, MLE: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data),
                loss_mle.item() / len(data)
            ))

    train_loss_epoch = train_loss / len(data_loader.dataset)
    train_loss_mle_epoch = train_loss_mle / len(data_loader.dataset)
    print('====> Epoch: {} Training set loss bce: {:.4f}, mle: {:.4f}'.format(
          epoch, train_loss_epoch, train_loss_mle_epoch))
    return train_loss_epoch, train_loss_mle_epoch


def test(epoch,data_loader,model,device,args):
    model.eval()
    test_loss = 0
    test_loss_mle = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            # add noise
            noisy_data = add_noise(data,device)
            noisy_data = noisy_data.to(device)
            recon_batch = model(noisy_data)
            test_loss += F.binary_cross_entropy(recon_batch, data.view(-1, 784))
            test_loss_mle += nn.MSELoss(reduction='sum')(recon_batch, data.view(-1, 784))
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        noisy_data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/%s_%s_rec_'  %(model.__name__,str(args.data_binary))+ str(epoch) + '.png', nrow=n)

    test_loss /= len(data_loader.dataset)
    test_loss_mle /= len(data_loader.dataset)

    print('====> Test set loss bce: {:.4f}, mle: {:.4f}'.format(test_loss,test_loss_mle))
