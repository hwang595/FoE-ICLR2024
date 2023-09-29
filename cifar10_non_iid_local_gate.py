# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from moe import MoE, MoELocal
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
parser.add_argument('--num_total_clients', type=int, help="number of total clients") # this is for testing how over method better leverages users' data
parser.add_argument('--alpha', type=float, help="alpha for non-iid split")
parser.add_argument('--fl_rounds', type=int, help="FL training rounds")
parser.add_argument('--local_bs', type=int, help="batch size for local training")
parser.add_argument('--local_ep', default=1, type=int, help="local training epochs")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for both local and global update")
parser.add_argument('--lr_decay_factor', default=0.995, type=float, help="learning rate decay factor.")
parser.add_argument('--momentum', default=0.5, type=float, help="momentum for both local and global update")
parser.add_argument('--test_bs', default=512, type=int, help="test batch size")
# MoE
parser.add_argument('--k', default=4, type=int, help="select top k")
# parser.add_argument('--num_experts', default=10, type=int, help="number of experts")
parser.add_argument('--hidden_size', default=256, type=int, help="hidden size")
parser.add_argument('--spreadout_regu', action='store_true',
                    help='apply spread-out regularization or not.')
parser.add_argument('--spreadout_opt_step', default=100, type=int,
                    help='the number of iterations to conduct global regularization steps.')
parser.add_argument('--spreadout_lr', default=0.0001, type=float, help="global spreadout regularization lr.")
args = parser.parse_args()


def spread_out_loss(w, args):
    loss = 0
    threshold = nn.Threshold(0, 0)
    for i in range(args.num_clients):
        sub_loss = 0
        for j in range(args.num_clients):
            if i != j:
                sub_loss += torch.dist(w[:,i], w[:,j], p=2)
        loss += threshold(10 - sub_loss) ** 2
    return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LocalUpdatePub(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                                                    num_workers=4,
                                                                    pin_memory=True)

    def train(self, net, client_idx, fl_round=0):
        # only update the client's local expert and gating function
        net.train()
        #optimizer = torch.optim.SGD(net.parameters(), 
        #                            lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(net.parameters(), 
                                   lr=self.args.lr, weight_decay=1e-4)      
        epoch_loss = []
        for epoch in range(args.local_ep):
            batch_loss = []
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logger.info('Client: {}, FL Round {:3d}, Local train loss {:.3f}, Num batches: {}'.format(
                    client_idx, fl_round, epoch_loss[-1], len(self.trainloader)))
        logger.info("!!! Gate w norm: {}".format(torch.norm(net.w_gate.data)))
        return sum(epoch_loss) / len(epoch_loss)


def global_aggregation(net_local_list, net_glob, args):
    global_feat_map_state_dict = {}
    # average feature map
    feat_map_buffer = [torch.zeros_like(v) for (k, v) in net_glob.feat_map.state_dict().items()]
    for local_net_idx, local_net in enumerate(net_local_list):
        for param_index, (k, v) in enumerate(local_net.feat_map.state_dict().items()):
            assert feat_map_buffer[param_index].size() == v.size()
            feat_map_buffer[param_index] += v
    # local averaged feature map to glboal model
    for param_index, (k, v) in enumerate(net_glob.feat_map.state_dict().items()):
        assert v.size() == feat_map_buffer[param_index].size()
        if "num_batches_tracked" in k:
           global_feat_map_state_dict[k] = (feat_map_buffer[param_index] / args.num_clients).long()
        else:
           global_feat_map_state_dict[k] = (feat_map_buffer[param_index] / args.num_clients)

    net_glob.feat_map.load_state_dict(global_feat_map_state_dict)

    # load experts state dict
    for local_id in range(args.num_clients):
        net_glob.experts[local_id].load_state_dict(net_local_list[local_id].expert.state_dict())

    for net_local_index, net_local in enumerate(net_local_list):
        net_glob.w_gate[:, net_local_index].data = net_local.w_gate[:, net_local_index].data
        net_glob.w_noise[:, net_local_index].data = net_local.w_noise[:, net_local_index].data

    if args.spreadout_regu:
        gate_optimizer = torch.optim.SGD([net_glob.w_gate], lr=args.spreadout_lr, momentum=0.9)
        for i in range(args.num_clients):
            sub_loss = 0
            for j in range(args.num_clients):
                if i != j:
                    sub_loss += torch.dist(net_glob.w_gate[:,i], net_glob.w_gate[:,j], p=2)
            logger.info("* sub_loss: {}".format(sub_loss))

        # global regularization step
        for i in range(args.spreadout_opt_step):
            gate_optimizer.zero_grad()
            loss = spread_out_loss(net_glob.w_gate, args)
            loss.backward()
            logger.info("@ iteration: {}, loss: {}".format(i, loss.item()))
            gate_optimizer.step()

        for i in range(args.num_clients):
            sub_loss = 0
            for j in range(args.num_clients):
                if i != j:
                    sub_loss += torch.dist(net_glob.w_gate[:,i], net_glob.w_gate[:,j], p=2)
            logger.info("@ sub_loss: {}".format(sub_loss))
    return


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
            #outputs = net.forward_expert(images, idx=0)
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
        
    logger.info("Dataset: {}; Global Valid. Acc.: {:.2f}".format(
            prefix, 100 * correct / total))


def test_local(net, testloader, prefix=''):
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
            outputs = net.forward_inf(images)
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
        
    logger.info("Dataset: {}; Local Valid. Acc.: {:.2f}".format(
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
testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=args.test_bs,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True)
# non-iid split for local udpate
train_id_list, train_cls_weight = cifar_noniid(trainset, args.num_total_clients, alpha=0.9)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# Take the number of experts the same as the number of clients
net_glob = MoE(num_classes=10, num_experts=args.num_clients, 
                hidden_size=args.hidden_size, noisy_gating=True, 
                k=args.k, device=device)
logger.info("@ Global MoE: {}, num_params: {}, global gate size: {}".format(net_glob, param_counter(net_glob), net_glob.w_gate.size()))
net_glob = net_glob.to(device)

net_local_list = [MoELocal(num_classes=10, num_experts=args.num_clients, 
                    hidden_size=args.hidden_size, noisy_gating=True, 
                    k=1, client_idx=client_index, device=device).to(device)
                for client_index in range(args.num_clients)
            ]
logger.info("@ Local MoE0: {}, num_params: {}, local gate size: {}".format(
                                            net_local_list[0], param_counter(net_local_list[0]),
                                            net_local_list[0].w_gate.size()))
net_glob.train()

for fl_round in range(args.fl_rounds):
    logger.info("FL round: {}, current lr: {}".format(fl_round, args.lr))

    for local_id in range(args.num_clients):
        local = LocalUpdatePub(args, trainset, train_id_list[local_id])
        loss = local.train(net=net_local_list[local_id], client_idx=local_id, fl_round=fl_round)
        if local_id == 0:
            test_local(net_local_list[local_id], testloader)

    _ = global_aggregation(net_local_list, net_glob, args=args)
    
    test(net_glob, testloader)
    args.lr = args.lr * args.lr_decay_factor
    
    for local_id in range(args.num_clients):
        net_local_list[local_id].feat_map.load_state_dict(net_glob.feat_map.state_dict())
        net_local_list[local_id].w_gate.data = net_glob.w_gate.data
        net_local_list[local_id].w_noise.data = net_glob.w_noise.data

logger.info('Finished Training')