#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import itertools

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random

import torch
from PIL import Image


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# class DatasetSplitMNIST(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset.data
#         self.targets = dataset.targets
#         self.idxs = idxs.numpy().tolist()

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item],:,:], self.targets[self.idxs[item]]
#         img = Image.fromarray(img.numpy(), mode="L")
#         return image, label


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_synthetic(dataset, no_participants):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_class = 10
    assert (no_participants == num_class)

    clas_weight = []
    per_participant_list = {}

    class_dominate_indices_list = []
    sub_indices_list = []

    # for class_index in range(num_class):
    #     label_indices = torch.where(dataset.targets == class_index)[0]
    #     per_participant_list[class_index] = label_indices
    _frac = 0.9
    # for each class, we partition _frac% of the data points to a class-dominate client
    # the rest (1-_frac)% data points are partitioned evenly to remaining clients
    for class_index in range(num_class):
        all_class_indices = np.where(np.array(dataset.targets) == class_index)[0].tolist()
        class_dominate_indices = all_class_indices[:int(_frac*len(all_class_indices))]
        sub_indices = np.array_split(all_class_indices[int(_frac*len(all_class_indices)):], no_participants-1)
        class_dominate_indices_list.append(class_dominate_indices)
        sub_indices_list.append(sub_indices)

    for client_index in range(no_participants):
        client_dominate_indices = class_dominate_indices_list[client_index]
        client_sub_indices = []
        for class_index in range(num_class):
            if client_index == class_index:
                continue
            elif client_index < class_index:
                client_sub_indices += sub_indices_list[class_index][client_index].tolist()
            else:
                client_sub_indices += sub_indices_list[class_index][client_index-1].tolist()
        per_participant_list[client_index] = client_dominate_indices + client_sub_indices

    return per_participant_list, clas_weight


def cifar_noniid(dataset, no_participants, alpha=0.9):
    """
    Input: Number of participants and alpha (param for distribution)
    Output: A list of indices denoting data in CIFAR training set.
    Requires: cifar_classes, a preprocessed class-indice dictionary.
    Sample Method: take a uniformly sampled 10-dimension vector as parameters for
    dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())
    class_size = len(cifar_classes[0])
    datasize = {}
    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i,j] for j in range(10)])
    clas_weight = np.zeros((no_participants,10))
    for i in range(no_participants):
        for j in range(10):
            clas_weight[i,j] = float(datasize[i,j])/float((train_img_size[i]))
    return per_participant_list, clas_weight


def cifar_synthetic(dataset, no_participants):
    """
    Input: Number of participants and alpha (param for distribution)
    Output: A list of indices denoting data in CIFAR training set.
    """
    num_class = 10
    assert (no_participants == num_class)

    clas_weight = []
    per_participant_list = {}

    class_dominate_indices_list = []
    sub_indices_list = []
    _frac = 0.75
    # for each class, we partition _frac% of the data points to a class-dominate client
    # the rest (1-_frac)% data points are partitioned evenly to remaining clients
    for class_index in range(num_class):
        all_class_indices = np.where(np.array(dataset.targets) == class_index)[0].tolist()
        class_dominate_indices = all_class_indices[:int(_frac*len(all_class_indices))]
        sub_indices = np.array_split(all_class_indices[int(_frac*len(all_class_indices)):], no_participants-1)
        class_dominate_indices_list.append(class_dominate_indices)
        sub_indices_list.append(sub_indices)

    for client_index in range(no_participants):
        client_dominate_indices = class_dominate_indices_list[client_index]
        client_sub_indices = []
        for class_index in range(num_class):
            if client_index == class_index:
                continue
            elif client_index < class_index:
                client_sub_indices += sub_indices_list[class_index][client_index].tolist()
            else:
                client_sub_indices += sub_indices_list[class_index][client_index-1].tolist()
        per_participant_list[client_index] = client_dominate_indices + client_sub_indices
    return per_participant_list, clas_weight


def cifar_coarse_label(dataset, no_participants=5):
    """
    this provides a coarse label based parition where classes that belong to the same coarse label will
    be partitioned to a client
    """
    num_super_classes = 20
    #assert (no_participants == num_super_classes)


    clas_weight = []
    per_participant_list = {}

    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                            6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                            5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                            10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                            16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    for client_index in range(no_participants):
        for sc_index in range(num_super_classes):
            # the i-th user gets the data points in the i-th subclass of all super classes
            sub_class_indices = np.where(coarse_labels == sc_index)[0]
            client_sub_class_indices = np.array_split(sub_class_indices, no_participants)[client_index]
            #all_sub_class_indices = np.where(np.array(dataset.targets) == sub_class_indices)[0].tolist()
            all_sub_class_indices = list(
                                        itertools.chain(*[np.where(np.array(dataset.targets) == csci)[0].tolist() for csci in client_sub_class_indices])
                                    )
            if client_index not in per_participant_list.keys():
                per_participant_list[client_index] = all_sub_class_indices
            else:
                per_participant_list[client_index].extend(all_sub_class_indices)

    return per_participant_list, clas_weight


def cifar_sample_label(dataset, no_participants=5, classes_per_parti=20):
    """
    this provides a coarse label based parition where classes that belong to the same coarse label will
    be partitioned to a client
    """
    # v 0.0 naive sampling
    # np.random.seed(666)
    # random.seed(666)

    # num_super_classes = 20

    # clas_weight = []
    # per_participant_list = {}

    # coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    #                         3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    #                         6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
    #                         0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    #                         5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
    #                         16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
    #                         10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
    #                         2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    #                         16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
    #                         18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    # client_sub_class_indices_collector = np.zeros((no_participants, classes_per_parti))
    # for client_index in range(no_participants):
    #     client_sub_class_indices = np.random.choice(coarse_labels.shape[0], classes_per_parti, replace=False)
    #     client_sub_class_indices_collector[client_index, :] = client_sub_class_indices
    #     print("@@@ sampled labels for client: {} are: {}".format(client_index, client_sub_class_indices))
    #     all_sub_class_indices = list(
    #                                 itertools.chain(*[np.where(np.array(dataset.targets) == csci)[0].tolist() for csci in client_sub_class_indices])
    #                             )
    #     if client_index not in per_participant_list.keys():
    #         per_participant_list[client_index] = all_sub_class_indices
    #     else:
    #         per_participant_list[client_index].extend(all_sub_class_indices)

    # common_count = np.zeros((no_participants, no_participants))
    # for i in range(no_participants):
    #     for j in range(no_participants):
    #         common_count[i][j] = np.intersect1d(client_sub_class_indices_collector[i,:], 
    #                                     client_sub_class_indices_collector[j,:]).shape[0]
    # print("** Class Common Count: {}".format(common_count))
    # return per_participant_list, client_sub_class_indices_collector

    # v 1.0 naive sampling
    num_super_classes = 20
    num_sub_classes = 100

    clas_weight = []
    per_participant_list = {}

    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                            6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                            5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                            10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                            16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

    client_sub_class_indices_collector = np.zeros((no_participants, classes_per_parti))
    for client_index in range(no_participants):
        client_sub_class_indices = []
        for class_count in range(classes_per_parti):
            if class_count < num_super_classes:
                # sample one subclass from each super classes
                sub_class_indices = np.where(coarse_labels == class_count)[0]
                sampled_sub_class_index = np.random.choice(sub_class_indices, size=1, replace=False)[0]
                client_sub_class_indices.append(sampled_sub_class_index)
                #print("class_count: {}, sub_class_indices: {}, sampled_sub_class_index: {}".format(
                #        class_count, sub_class_indices, sampled_sub_class_index
                #    ))
            else:
                # sample one super class
                # sample on subclass in that super class which has not been sampled yet
                #sampled_sup_class_index = np.random.choice(num_super_classes, size=1, replace=False)[0]
                #sub_class_indices = np.where(coarse_labels == sampled_sup_class_index)[0]
                #already_sampled_subclass = np.intersect1d(np.arange(num_sub_classes), client_sub_class_indices)
                #remaining_sub_class_indices = np.delete(sub_class_indices, np.where(sub_class_indices == already_sampled_subclass)[0][0])
                #sampled_sub_class_index = np.random.choice(remaining_sub_class_indices, size=1, replace=False)[0]
                #sampled_sup_class_index = np.random.choice(num_super_classes, size=1, replace=False)[0]
                remaining_sub_classes = []
                for sc_index in range(num_sub_classes):
                    if sc_index not in client_sub_class_indices:
                        remaining_sub_classes.append(sc_index)
                client_sub_class_indices += np.random.choice(remaining_sub_classes, classes_per_parti-num_super_classes, replace=False).tolist()
                break
        client_sub_class_indices_collector[client_index, :] = client_sub_class_indices

        print("@@@ sampled labels for client: {} are: {}".format(client_index, client_sub_class_indices))
        all_sub_class_indices = list(
                                    itertools.chain(*[np.where(np.array(dataset.targets) == csci)[0].tolist() for csci in client_sub_class_indices])
                                )
        if client_index not in per_participant_list.keys():
            per_participant_list[client_index] = all_sub_class_indices
        else:
            per_participant_list[client_index].extend(all_sub_class_indices)

    common_count = np.zeros((no_participants, no_participants))
    for i in range(no_participants):
        for j in range(no_participants):
            common_count[i][j] = np.intersect1d(client_sub_class_indices_collector[i,:], 
                                        client_sub_class_indices_collector[j,:]).shape[0]
    print("** Class Common Count: {}".format(common_count))
    return per_participant_list, client_sub_class_indices_collector

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)