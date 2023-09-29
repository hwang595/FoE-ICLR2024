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

#from utils.sampling import *
from datasets.fed_cifar100.data_loader import load_partition_data_federated_cifar100
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
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for both local and global update")
parser.add_argument('--lr_decay_factor', default=0.995, type=float, help="learning rate decay factor.")
parser.add_argument('--momentum', default=0.5, type=float, help="momentum for both local and global update")
parser.add_argument('--test_bs', default=512, type=int, help="test batch size")
parser.add_argument('--local_log_freq', default=5, type=int, help="frequency of local logging")
# MoE
parser.add_argument('--k', default=4, type=int, help="select top k")
# parser.add_argument('--num_experts', default=10, type=int, help="number of experts")
parser.add_argument('--hidden_size', default=256, type=int, help="hidden size")

args = parser.parse_args()


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
    def __init__(self, args, dataloader=None):
        if args.dataset == "cifar10":
            num_classes = 10
        elif args.dataset == "cifar100":
            num_classes = 100

        self.args = args
        self.criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
        self.trainloader = dataloader

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
            if epoch % args.local_log_freq == 0:
                logger.info('Client: {}, FL Round {:3d}, Local train loss {:.3f}'.format(client_idx, fl_round, epoch_loss[-1]))
        #logger.info("!!! Gate w norm: {}".format(torch.norm(net.w_gate.data)))
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
    #net_glob.feat_map.load_state_dict(net_local_list[0].feat_map.state_dict())

    # load experts state dict
    for local_id in range(args.num_clients):
        net_glob.experts[local_id].load_state_dict(net_local_list[local_id].expert.state_dict())

    for net_local_index, net_local in enumerate(net_local_list):
        net_glob.w_gate[:, net_local_index].data = net_local.w_gate[:, net_local_index].data
        net_glob.w_noise[:, net_local_index].data = net_local.w_noise[:, net_local_index].data
    return


def test(net, testloader, prefix=''):
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
        
    logger.info("Dataset: {}; Global Valid. Acc.: {:.2f}".format(
            prefix, 100 * correct / total))


def test_local(net, testloader, prefix=''):
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
        
    logger.info("Dataset: {}; Local Valid. Acc.: {:.2f}".format(
            prefix, 100 * correct / total))


# # non-iid split for local udpate
# train_id_list, train_cls_weight = cifar_noniid(trainset, args.num_clients, alpha=0.9)
DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = load_partition_data_federated_cifar100(
                                                                                    data_dir="datasets/fed_cifar100/datasets", batch_size=args.local_bs)
#print(len(train_data_local_dict), len(test_data_local_dict))

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# Take the number of experts the same as the number of clients
net_glob = MoE(num_classes=100, num_experts=args.num_clients, 
                hidden_size=args.hidden_size, noisy_gating=True, 
                k=args.k, device=device)
logger.info("@ Global MoE: {}, num_params: {}, global gate size: {}".format(net_glob, param_counter(net_glob), net_glob.w_gate.size()))
net_glob = net_glob.to(device)

net_local_list = [MoELocal(num_classes=100, num_experts=args.num_clients, 
                    hidden_size=args.hidden_size, noisy_gating=True, 
                    k=1, client_idx=client_index, device=device).to(device)
                for client_index in range(args.num_clients)
            ]
logger.info("@ Local MoE0: {}, num_params: {}, local gate size: {}".format(net_local_list[0], param_counter(net_local_list[0]),
                                                                    net_local_list[0].w_gate.size()))
net_glob.train()

for fl_round in range(args.fl_rounds):
    logger.info("FL round: {}, current lr: {}".format(fl_round, args.lr))

    for local_id in range(args.num_clients):
        local = LocalUpdatePub(args, train_data_local_dict[local_id])
        loss = local.train(net=net_local_list[local_id], client_idx=local_id, fl_round=fl_round)
        if local_id == 0:
            test_local(net_local_list[local_id], test_data_global)

    _ = global_aggregation(net_local_list, net_glob, args=args)
    
    test(net_glob, test_data_global)
    args.lr = args.lr * args.lr_decay_factor
    
    for local_id in range(args.num_clients):
        net_local_list[local_id].feat_map.load_state_dict(net_glob.feat_map.state_dict())
        net_local_list[local_id].w_gate.data = net_glob.w_gate.data
        net_local_list[local_id].w_noise.data = net_glob.w_noise.data

logger.info('Finished Training')