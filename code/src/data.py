import torchvision
import os
import torch
from typing import Optional

#data directory, using os to get the current directory
def get_data_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__)) # should be util
    #directory one level above
    home_dir = os.path.dirname(cur_dir)
    #check if the data directory is exist
    if not os.path.exists(os.path.join(home_dir, 'data')):
        os.makedirs(os.path.join(home_dir, 'data'))
    #assign the data directory
    data_dir = os.path.join(home_dir, 'data') # this is to get the data directory
    return data_dir

# create class data set for different input

class Dataset():
    def __init__(self, data_code):
        self.data = data_code.lower()
        self.data_train: Optional[torch.utils.data.DataSet] = None
        self.data_test: Optional[torch.utils.data.DataSet] = None
        self.data_dir = get_data_dir()

    def generate(self):
        if self.data in ['mnist','cifar10']:
            if self.data == 'mnist':
                torchvision.datasets.MNIST(root=self.data_dir, train=True, 
    transform=torchvision.transforms.ToTensor(), download=True)
                torchvision.datasets.MNIST(root=self.data_dir, train=False, 
    transform=torchvision.transforms.ToTensor(), download=True)
                return print('MNIST data downloaded and saved in ' + self.data_dir)
            elif self.data == 'cifar10':
                return print('Not implemented yet')
        else:
            return print('Data not found')
    
    
    def train_loader(self, batch_size):
        if self.data == 'mnist':
            self.data_train = torchvision.datasets.MNIST(root=self.data_dir, train=True, transform = torchvision.transforms.ToTensor(), download=False)
        
        return torch.utils.data.DataLoader(self.data_train, batch_size=batch_size, shuffle=True, num_workers=0)
    
    def test_loader(self, batch_size):
        if self.data == 'mnist':
            self.data_test = torchvision.datasets.MNIST(root=self.data_dir, train=False, transform = torchvision.transforms.ToTensor(), download=False)
        
        return torch.utils.data.DataLoader(self.data_test, batch_size=batch_size, shuffle=False, num_workers=0)
                                               
#if __name__ == '__main__':
#    _ = Dataset()


# create class model for different input
#data_mnist = Dataset('mnist')
#data_mnist.generate()