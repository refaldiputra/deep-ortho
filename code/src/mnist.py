# This is adopted from https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/master
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from PIL import Image

# pre-processing
def global_contrast_normalization(x, scale):
    """Apply global contrast normalization to tensor. """
    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale
    return x

def get_data_dir(): # for the data directory
    cur_dir = os.path.dirname(os.path.abspath(__file__)) # should be util
    #directory one level above
    home_dir = os.path.dirname(cur_dir)
    #check if the data directory is exist
    if not os.path.exists(os.path.join(home_dir, 'data')):
        os.makedirs(os.path.join(home_dir, 'data'))
    #assign the data directory
    data_dir = os.path.join(home_dir, 'data') # this is to get the data directory
    return data_dir


class MNIST_loader(data.Dataset):
    """Creating class for the dataset"""
    def __init__(self, data, target, transform):
        self.data = data # our input data
        self.target = target # our label (0 or 1)
        self.transform = transform # the transformation

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index] # get the input data
        y = self.target[index] # get the label
        if self.transform: # apply the transformation
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        """number of samples."""
        return len(self.data)

class MNIST():
    def __init__(self, normal, batch_size, data_dir=get_data_dir()):
        self.normal = normal
        self.batch_size = batch_size
        self.data_dir = data_dir
    def get_mnist(self):
        # We want to select i-th class as normal
        # we also want to manipulate the batch size
        """get dataloders"""
        # min, max values for each class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                    (-0.6661464580883915, 20.108062262467364),
                    (-0.7820454743183202, 11.665100841080346),
                    (-0.7645772083211267, 12.895051191467457),
                    (-0.7253923114302238, 12.683235701611533),
                    (-0.7698501867861425, 13.103278415430502),
                    (-0.778418217980696, 10.457837397569108),
                    (-0.7129780970522351, 12.057777597673047),
                    (-0.8280402650205075, 10.581538445782988),
                    (-0.7369959242164307, 10.697039838804978)]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale = 'l2')),
                                        transforms.Normalize([min_max[self.normal][0]],
                                                            [min_max[self.normal][1] \
                                                            -min_max[self.normal][0]])])
        train = datasets.MNIST(root=self.data_dir, train=True, download=True)
        test = datasets.MNIST(root=self.data_dir, train=False, download=True)

        x_train = train.data
        y_train = train.targets

        x_train = x_train[np.where(y_train==self.normal)] # focus on the i-th class as normal
        y_train = y_train[np.where(y_train==self.normal)] # focus on the i-th class as normal
                                        
        data_train = MNIST_loader(x_train, y_train, transform)
        dataloader_train = DataLoader(data_train, batch_size=self.batch_size, 
                                    shuffle=True, num_workers=0)
        
        x_test = test.data
        y_test = test.targets
        y_test = np.where(y_test==self.normal, 0, 1)
        data_test = MNIST_loader(x_test, y_test, transform)
        dataloader_test = DataLoader(data_test, batch_size=self.batch_size, 
                                    shuffle=True, num_workers=0)
        return dataloader_train, dataloader_test # directly call the loaders