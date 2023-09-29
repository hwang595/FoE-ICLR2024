import itertools
import json
import random
import os

import logging
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from .sampling import DatasetSplit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CIFAR100_COARSE_LABELS = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                        3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                        6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                        0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                        5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                        16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                        10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                        2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                        16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                        18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
CIFAR100_SUBCLASSES = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                      ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                      ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                      ['bottle', 'bowl', 'can', 'cup', 'plate'],
                      ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                      ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                      ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                      ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                      ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                      ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                      ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                      ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                      ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                      ['crab', 'lobster', 'snail', 'spider', 'worm'],
                      ['baby', 'boy', 'girl', 'man', 'woman'],
                      ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                      ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                      ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                      ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                      ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
# from https://gist.github.com/adam-dziedzic/4322df7fc26a1e75bee3b355b10e30bc
CIFAR100_FINE_LABELS = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]


def find_nearest_indices(reference_vector, samples, N=1):
    # written by GPT-4 :P
    # Calculate the squared differences
    diff = samples - reference_vector
    squared_diff = np.square(diff)
    
    # Sum along axis 1 to get the squared distance
    squared_distance = np.sum(squared_diff, axis=1)
    
    # Sort and get the N nearest indices
    nearest_indices = np.argsort(squared_distance)[:N]
    
    return nearest_indices


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


class SimpleClassifier(nn.Module):
    def __init__(self, num_clients, num_classes, hidden_dim=512):
        super(SimpleClassifier, self).__init__()

        self.fc1 = nn.Linear(int(num_clients*num_classes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class SimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, targets, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img, target = self.data[idx, :], self.targets[idx]
        return img, target

def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Seeded everything")


def get_loaders(dataset, idxs, valid=False, args=None):
    if valid:
        num_valid = 500
        idxs = np.random.permutation(idxs)
        train_idxs = idxs[:len(idxs)-500]
        valid_idxs = idxs[len(idxs)-500:]
    
        valoader = torch.utils.data.DataLoader(DatasetSplit(dataset, valid_idxs), batch_size=args.test_bs, 
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True,
                                                                drop_last=True)
    else:
        valoader, valid_idxs = None, None
    trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset, train_idxs), batch_size=args.local_bs, 
                                                                shuffle=True,
                                                                num_workers=4,
                                                                pin_memory=True,
                                                                drop_last=True)

    return trainloader, valoader, valid_idxs

def save_network_logits(net, data_loader, file_name, args=None):
    net_logits_collector = []
    labes_collector = []
    device = get_device()
    for i, data in enumerate(data_loader):
        inputs, labels = data

        # we conduct coarse label classification here
        labels = CIFAR100_COARSE_LABELS[labels]
        inputs = inputs.to(device)
        outputs = net(inputs)
        net_logits_collector.append(outputs.data.cpu())
        labes_collector.append(labels)

    dataset_logits = torch.cat(net_logits_collector).numpy()
    dataset_labels = np.concatenate(labes_collector)

    logger.info("Saving logits, dim: {}, Labels dim: {}".format(dataset_logits.shape,
                        dataset_labels.shape))
    dict_to_save = {"logits":dataset_logits, "labels":dataset_labels}

    with open(file_name+".npy","wb") as file_to_save:
        np.save(file_to_save, dict_to_save)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return device


def get_dataset(ds=None, validation_split=False, valid_size=5000):
    if ds == "cifar10":
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        nb_classes = 10
        ds = "CIFAR10"
    elif ds == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]
        nb_classes = 20 # we used this dataset specifically for superclass classification
        ds = "CIFAR100"
    else:
        raise AttributeError

    transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = getattr(torchvision.datasets, ds)(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
    testset = getattr(torchvision.datasets, ds)(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transform_test)
    if validation_split:
        valset = getattr(torchvision.datasets, ds)(root='./data',
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)
        indices = torch.randperm(len(trainset))
        train_indices = indices[:len(indices) - valid_size].numpy()
        valid_indices = indices[len(indices) - valid_size:].numpy() if valid_size else None

        trainset.data = trainset.data[train_indices, :, :, :]
        trainset.targets = np.array(trainset.targets)[train_indices]
        valset.data = valset.data[valid_indices, :, :, :]
        valset.targets = np.array(valset.targets)[valid_indices]
        return trainset, testset, valset, nb_classes
    return trainset, testset, nb_classes


def topk_indices(arr, k):
    # Returns the indices of the top-k largest values
    # generated by GPT :P
    return np.argsort(arr)[-k:][::-1]