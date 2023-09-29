# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn as nn
from moe import MoE
import torch.optim as optim

import argparse
import copy
import itertools
import logging

from utils.sampling import *
from utils.network_utils import param_counter

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Non-iid CIFAR FL-MoE Training with public dataset to train gating.')
# FL
parser.add_argument('--num_clients', type=int, help="number of clients")
parser.add_argument('--alpha', type=float, help="alpha for non-iid split")
parser.add_argument('--fl_rounds', type=int, help="FL training rounds")
parser.add_argument('--local_bs', type=int, help="batch size for local training")
parser.add_argument('--local_ep', default=1, type=int, help="local training epochs")
parser.add_argument('--global_bs', type=int, help="batch size for global update")
parser.add_argument('--global_ep', default=1, type=int, help="global update epoch")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for both local and global update")
parser.add_argument('--lr_decay_factor', default=0.99, type=float, help="learning rate decay factor.")
parser.add_argument('--momentum', default=0.5, type=float, help="momentum for both local and global update")
parser.add_argument('--test_bs', default=512, type=int, help="test batch size")
# MoE
parser.add_argument('--k', default=4, type=int, help="select top k")
# parser.add_argument('--num_experts', default=10, type=int, help="number of experts")
parser.add_argument('--hidden_size', default=256, type=int, help="hidden size")

args = parser.parse_args()


class LocalUpdatePub(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                                                    num_workers=4,
                                                                    pin_memory=True)

    def train(self, net, client_idx, fl_round=0):
        # only update the client's local expert
        params = [net.experts[client_idx].parameters(), net.feat_map.parameters()]
        optimizer = torch.optim.SGD(itertools.chain(*params), 
                                    lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-4)
        epoch_loss = []
        for epoch in range(args.local_ep):
            batch_loss = []
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net.forward_expert(inputs, client_idx)

                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logger.info('Client: {}, FL Round {:3d}, Local train loss {:.3f}'.format(client_idx, fl_round, epoch_loss[-1]))
        return net.feat_map.state_dict(), sum(epoch_loss) / len(epoch_loss)


class GlobalUpdatePub(object):
    def __init__(self, args, trainloader=None, testloader=None):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, net):
        # global update only updates gating and noise parameters
        params = [net.w_gate]+[net.w_noise]
        optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-4)

        epoch_loss = []
        for _ in range(args.global_ep):
            batch_loss = []
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs, aux_loss = net(inputs)

                loss = self.criterion(outputs, labels)
                total_loss = loss + aux_loss
                total_loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

    def test(self, net):
        test(net, self.testloader, 'CIFAR10')


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test(net, testloader, prefix=''):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        logger.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
    logger.info("Dataset: {}; Valid. Acc.: {:.2f}".format(
            prefix, 100 * correct / total))


def get_dataset(ds=None):
    if ds == "CIFAR10":
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    elif ds == "CIFAR100":
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]
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

    return trainset, testset


trainset, testset = get_dataset("CIFAR10")
trainloader = torch.utils.data.DataLoader(trainset,
                                        batch_size=args.global_bs,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True)
testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=args.test_bs,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True)
# non-iid split for local udpate
train_id_list, train_cls_weight = cifar_noniid(trainset, args.num_clients, alpha=0.9)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# Take the number of experts the same as the number of clients
net_glob = MoE(num_classes=10, num_experts=args.num_clients, 
                hidden_size=args.hidden_size, noisy_gating=True, 
                k=args.k, device=device)
logger.info("@ Global MoE: {}, num_params: {}".format(net_glob, param_counter(net_glob)))
net_glob = net_glob.to(device)

net_local_list = [MoE(num_classes=10, num_experts=args.num_clients, 
                    hidden_size=args.hidden_size, noisy_gating=True, 
                    k=args.k, device=device).to(device)
                for _ in range(args.num_clients)
            ]

net_glob.train()
global_feat_map = net_glob.feat_map.state_dict()
global_gate = net_glob.w_gate.data
global_noise = net_glob.w_noise.data

for fl_round in range(args.fl_rounds):
    logger.info("FL round: {}, current lr: {}".format(fl_round, args.lr))
    w_locals, loss_locals = [], []
    for local_id in range(args.num_clients):
        local = LocalUpdatePub(args, trainset, train_id_list[local_id])
        
        # load global gate and noise
        net_local_list[local_id].feat_map.load_state_dict(global_feat_map)
        net_local_list[local_id].w_gate.data = global_gate
        net_local_list[local_id].w_noise.data = global_noise

        w_fm, loss = local.train(net=net_local_list[local_id], client_idx=local_id, fl_round=fl_round)
        w_locals.append(w_fm)
        loss_locals.append(loss)
    loss_avg = sum(loss_locals) / len(loss_locals)

    avg_feat_map = FedAvg(w_locals)

    # global network load averaged feature map
    net_glob.feat_map.load_state_dict(avg_feat_map)
    for local_id in range(args.num_clients):
        net_glob.experts[local_id].load_state_dict(net_local_list[local_id].experts[local_id].state_dict())
    
    glob = GlobalUpdatePub(args, trainloader, testloader)
    glob.train(net_glob)
    glob.test(net_glob)
    global_feat_map = net_glob.feat_map.state_dict()
    global_gate = net_glob.w_gate.data
    global_noise = net_glob.w_noise.data
    args.lr = args.lr * args.lr_decay_factor

logger.info('Finished Training')