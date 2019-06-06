import sys
import torch
from torchvision import transforms
from torchvision.utils import save_image

def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))

r"""
Binarized data is obtained by min_max+round
"""
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)

def binarize_data():

    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))])


def add_noise(img,device):
    noise = torch.randn(img.size()) * 0.4
    noise = noise.to(device)
    noisy_img = img + noise
    return noisy_img
