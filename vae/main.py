from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torchvision import datasets, transforms
from ae_utils import add_noise, binarize_data
from autoencoders import DAE,VAE,AE
from train_test import train,test
import logging
from log_utils import Logger


parser = argparse.ArgumentParser(description='AE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data_binary', type=int, default=0, metavar='S',
                    help='binarize data (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda:3" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# binarize ver
if args.data_binary:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=binarize_data()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=binarize_data()),
        batch_size=args.batch_size, shuffle=True, **kwargs)


# ori ver
if not args.data_binary:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)


model_to_use = [
    DAE,
    # AE,
    # VAE
]

record_t_e = []
record_t_mle_e = []
for Net in model_to_use:
    print('-----\n{}'.format(Net.__name__))
    # f = open('%s.log' %(Net.__name__), 'a')
    # sys.stdout = f
    # sys.stderr = f
    # sys.stdout = Logger('%s.log' %(Net.__name__), sys.stdout)
    # sys.stderr = Logger('%s.log_file' %(Net.__name__), sys.stderr)
    logger = Logger(log_file_name='%s.txt' %(Net.__name__),
    log_level=logging.DEBUG, logger_name='%s' %(Net.__name__)).get_log()
    model = Net().to(device)
    for epoch in range(1, args.epochs + 1):
        [t_e,t_mle_e] = train(epoch,train_loader,model,device,args)
        record_t_e.append(t_e)
        record_t_mle_e.append(t_mle_e)
        test(epoch,test_loader,model,device,args)

import matplotlib.pyplot as plt
import numpy as np
x_axis = np.arange(1,args.epochs+1)
fig_te = plt.figure()
plt.plot(x_axis,record_t_e)
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.show()
plt.savefig('results/%s_%s_rec' % (Net.__name__, str(args.data_binary)) + '.png')



plt.figure()
plt.plot(x_axis,record_t_mle_e)
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.show()
plt.savefig('results/%s_%s_rec' % (Net.__name__, str(args.data_binary)) + '.png')




