# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from moe import MoE
import torch.optim as optim

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
trainloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=128,
                                          shuffle=True, 
                                          num_workers=4,
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False,
                                       download=True, 
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, 
                                        batch_size=128,
                                        shuffle=False, 
                                        num_workers=4,
                                        pin_memory=True)


if torch.cuda.is_available():
    device = torch.device('cuda:2')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

#net = MoE(input_size=3072,output_size= 10, num_experts=10, hidden_size=256, noisy_gating=True, k=4, device=device)
net = MoE(num_classes=10, num_experts=10, hidden_size=256, noisy_gating=True, k=4, device=device)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)

net.train()
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #inputs = inputs.view(inputs.shape[0], -1)
        outputs, aux_loss = net(inputs)
        loss = criterion(outputs, labels)
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('# [Epoch: %d, Iter: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


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

# yields a test accuracy of around 39 %