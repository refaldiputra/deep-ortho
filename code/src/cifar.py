## adopted from https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/master
from torch.utils.data import Subset, DataLoader
from PIL import Image
import torch
import numpy as np
from torchvision.datasets import CIFAR10
import os
import torchvision.transforms as transforms

# pre-processing <-- should be moved to utils
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

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    This function is to assign the index of the target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()

def get_data_dir(): # for the data directory
    cur_dir = os.path.dirname(os.path.abspath(__file__)) # should be util
    #directory one level above
    home_dir = os.path.dirname(cur_dir)
    #check if the data directory is exist
    if not os.path.exists(os.path.join(home_dir, 'data/CIFAR10')): #customize to cifar
        os.makedirs(os.path.join(home_dir, 'data/CIFAR10'))
    #assign the data directory
    data_dir = os.path.join(home_dir, 'data/CIFAR10') # this is to get the data directory
    return data_dir

class CIFAR10_Dataset():
    def __init__(self, normal, data_dir=get_data_dir()):
        super(CIFAR10_Dataset).__init__()

        self.n_classes = 2  # 0: normal, 1: outlier <-- it means we focus on one class and treat the rest as outliers
        self.normal_classes = tuple([normal]) # focus only on normal class
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal) # the rest are treated as outliers

        # Pre-computed min and max values from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]

        # CIFAR-10 preprocessing: can be l1 or l2
        # The transformation for the training and test are different.
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal][0]] * 3,
                                                             [min_max[normal][1] - min_max[normal][0]] * 3)]
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
        #                      std=[0.247, 0.243, 0.261])]
        )

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        train_set = MyCIFAR10(root=data_dir, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # print(train_set)
        # Subset train set to normal class
        # This is a nice way to get the label and its index when we subset the normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes) 
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCIFAR10(root=data_dir, train=False, download=True,
                                  transform=transform, target_transform=target_transform)
        
    def get_loaders(self,batch_size) -> (DataLoader, DataLoader):
        # data loader (self.train_loader, self.test_loader)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, 
        shuffle=False, num_workers=0)
        return self.train_loader, self.test_loader


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        # print(self.train)
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index  # only line changed