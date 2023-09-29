# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from moe import MoE
import torch.optim as optim

import argparse
import copy

from utils.sampling import *


parser = argparse.ArgumentParser(description='Non-iid CIFAR FL-MoE Training.')
# FL
parser.add_argument('--num_clients', type=int, help="number of clients")
parser.add_argument('--alpha', type=float, help="alpha for non-iid split")
parser.add_argument('--epochs', type=int, help="global training epochs")
parser.add_argument('--local_bs', type=int, help="batch size for local training")
parser.add_argument('--local_ep', default=1, type=int, help="local training epochs")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
# MoE
parser.add_argument('--k', default=4, type=int, help="select top k")
parser.add_argument('--num_experts', default=10, type=int, help="number of experts")
parser.add_argument('--hidden_size', default=256, type=int, help="hidden size")

args = parser.parse_args()


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for _ in range(args.local_ep):
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
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test(net, testloader):
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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# data prep for test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=128,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True)

# non-iid split
train_id_list, train_cls_weight = cifar_noniid(trainset, args.num_clients, alpha=0.9)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

net_glob = MoE(num_classes=10, num_experts=args.num_experts, hidden_size=args.hidden_size, noisy_gating=True, k=args.k, device=device)
net_glob = net_glob.to(device)
net_glob.train()

w_glob = net_glob.state_dict()

for epoch in range(args.epochs):
    w_locals, loss_locals = [], []
    for local_id in range(args.num_clients):
        local = LocalUpdate(args, trainset, train_id_list[local_id])
        net_glob.load_state_dict(w_glob)
        w, loss = local.train(net=net_glob)
        w_locals.append(w)
        loss_locals.append(loss)

    w_glob = FedAvg(w_locals)

    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Train loss {:.3f}'.format(epoch, loss_avg))
    test(net_glob, testloader)

print('Finished Training')

# yields a test accuracy of around 39 %