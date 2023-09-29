import logging
import itertools
import pickle
import random

import torch
import numpy as np

from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader

import copy
import itertools
import sys
sys.path.append("..")
from utils.utils import get_device, \
CIFAR100_COARSE_LABELS, CIFAR100_SUBCLASSES, CIFAR100_FINE_LABELS, \
SimpleClassifier, SimpleDataset, findsubsets, find_nearest_indices, topk_indices
from utils.local_uncertainty import to_np

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def cross_expert_accs(net_local_list, test_loader, client_indices_collector, args):
    cross_expert_accs = np.zeros((args.num_clients, args.num_clients))
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100

    device = get_device()

    for client_i in range(args.num_clients):
        net = net_local_list[client_i]
        for client_j in range(args.num_clients):
            correct = 0
            total = 0
            
            if args.par_strategy == "corse-label":
                sub_class_indices = [np.array_split(np.where(
                    CIFAR100_COARSE_LABELS == sc_index)[0], args.num_clients)[client_j].tolist() for sc_index in range(20)]
                client_sub_classes = list(itertools.chain(*sub_class_indices))
                client_sub_labels = [CIFAR100_SUBCLASSES[sc_index][client_j] for sc_index in range(20)]
            elif args.par_strategy == "sample-based":
                client_sub_classes = client_indices_collector[client_j, :]
                client_sub_labels = [CIFAR100_FINE_LABELS[int(cci)] for cci in client_indices_collector[client_j, :]]
            else:
                raise NotImplementedError("Unsupported par strategy ...")
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data

                    super_labels = torch.from_numpy(CIFAR100_COARSE_LABELS[labels]).to(device)
                    L = super_labels

                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)

                    _, predicted = torch.max(outputs.data, 1)

                    # collect the correct predictions for each class
                    for label, ll, prediction in zip(L, labels, predicted):
                        if ll.item() in client_sub_classes:
                            if label == prediction:
                                correct += 1
                            total += 1
            cross_expert_accs[client_i][client_j] = float(correct/total) * 100.0
    logger.info("Corss Exper Accs: {}".format(cross_expert_accs))


def cross_expert_confs(net_local_list, train_loaders, args):
    if args.use_histgram:
        cross_expert_confs = []
    else:
        cross_expert_confs = np.empty((args.num_clients, args.num_clients))
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100

    device = get_device()

    for client_i in range(args.num_clients):
        net = net_local_list[client_i]
        if args.use_histgram:
            tempt_cross_expert_confs = []
        for client_j in range(args.num_clients):
            if args.use_histgram:
                conf_score = []
            else:
                conf_score = 0
                total = 0
            
            with torch.no_grad():
                for data in train_loaders[client_j]:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)

                    output = net(images)
                    smax = to_np(F.softmax(output, dim=1))

                    if args.e_mat_score == "maxsoftmax":
                        scores = np.max(smax, axis=1)
                    elif args.e_mat_score == "entropy":
                        scores = entropy(smax, axis=1)
                    else:
                        raise NotImplementedError("Unsupported E mat Score ...")

                    if args.use_histgram:
                        conf_score.append(scores)
                    else:
                        conf_score += np.sum(scores)
                        total += scores.shape[0]

            if args.use_histgram:
                tempt_histgram = np.histogram(conf_score, bins=10)
                tempt_histgram = (tempt_histgram[0]/tempt_histgram[0].sum(), tempt_histgram[1])
                tempt_cross_expert_confs.append(tempt_histgram)
            else:
                cross_expert_confs[client_i][client_j] = float(conf_score/total)

        if args.use_histgram:
            cross_expert_confs.append(tempt_cross_expert_confs)
    logger.info("Corss Exper Confs: {}".format(cross_expert_confs))
    return cross_expert_confs


def cross_expert_classifier(net_local_list, trainloader, client_class_indices, args):
    if args.use_histgram:
        cross_expert_confs = []
    else:
        cross_expert_confs = np.empty((args.num_clients, args.num_clients))
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100

    device = get_device()
    if args.pred_mode == "label-pred":
        _classification_class = 20
    elif args.pred_mode == "expert-pred":
        _classification_class = args.num_clients
    else:
        raise NotImplementedError("Unsupported prediction mode ...")

    classifier = SimpleClassifier(num_clients=args.sampled_experts, num_classes=_classification_class).to(device)

    new_dataset = np.zeros((len(trainloader.dataset), int(args.num_clients * args.sampled_experts)))
    new_labels = np.zeros(len(trainloader.dataset))
    sampled_expert_list = np.random.choice(args.num_clients, args.sampled_experts, replace=False)
    print("** sampled experts: {}".format(sampled_expert_list))

    for batch_idx, data in enumerate(trainloader):
        with torch.no_grad():
            images, ori_labels = data
            labels = CIFAR100_COARSE_LABELS[ori_labels]
            images = images.to(device)
            # constructing datasets
            client_shift_idx = 0
            for client_i in range(args.num_clients):
                if client_i in sampled_expert_list:
                    net = net_local_list[client_i]
                    output = net(images)
                    smax = to_np(F.softmax(output, dim=1))
                    new_dataset[args.local_bs*batch_idx:args.local_bs*(batch_idx+1), 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                    if args.pred_mode == "label-pred":
                        new_labels[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = labels
                    elif args.pred_mode == "expert-pred":
                        expert_labels = np.zeros(labels.shape)
                        for l_idx, ll in enumerate(ori_labels):
                            tempt_label = []
                            for client_index in range(args.num_clients):
                                if ll.item() in client_class_indices[client_index, :].tolist():
                                    tempt_label.append(client_index)
                            #print("** ll: {}, tempt_label: {}".format(ll, tempt_label))
                            expert_labels[l_idx] = np.random.choice(tempt_label, size=1, replace=False)
                        new_labels[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = expert_labels
                    else:
                        raise NotImplementedError("Unsupported prediction mode ...")
                    client_shift_idx += 1

    _new_classifier_eps = 50
    new_data_loader = torch.utils.data.DataLoader(SimpleDataset(data=new_dataset, targets=new_labels),
                                                    batch_size=128, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_classifier_eps)

    classifier.train()
    train_loss = 0
    correct = 0
    total = 0
    for ep in range(_new_classifier_eps):
        for batch_idx, (inputs, targets) in enumerate(new_data_loader):
            inputs, targets = inputs.float().to(device), targets.long().to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                logger.info("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                        ep,
                        batch_idx, len(trainloader), train_loss/(batch_idx+1), 
                        100.*correct/total, correct, total))
        scheduler.step()
    return classifier, sampled_expert_list


def oracle_layer(net_local_list, trainloader, testloader, layer_idx, cur_exps_list, args, device):
    # we only support label pred now:
    _classification_class = 20

    final_accs = []
    for expert_index in range(args.num_clients):
        if expert_index in cur_exps_list: # to make sure client in `cur_exps_list` will always be considered
            final_accs.append(0.0)
            continue
        else:
            temp_exps_list = cur_exps_list + [expert_index] 
        classifier = SimpleClassifier(num_clients=layer_idx, num_classes=_classification_class).to(device)
        new_dataset = np.zeros((len(trainloader.dataset), int(_classification_class * layer_idx)))
        new_labels = np.zeros(len(trainloader.dataset))
        for batch_idx, data in enumerate(trainloader):
            with torch.no_grad():
                images, ori_labels = data
                labels = CIFAR100_COARSE_LABELS[ori_labels]
                images = images.to(device)
                # constructing datasets
                client_shift_idx = 0
                for client_i in range(args.num_clients):
                    if client_i in temp_exps_list:
                        net = net_local_list[client_i]
                        output = net(images)
                        smax = to_np(F.softmax(output, dim=1))
                        new_dataset[args.local_bs*batch_idx:args.local_bs*(batch_idx+1), 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                        new_labels[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = labels
                        client_shift_idx += 1
        # train a model
        _new_classifier_eps = 50
        new_data_loader = torch.utils.data.DataLoader(SimpleDataset(data=new_dataset, targets=new_labels),
                                                        batch_size=128, shuffle=True, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_classifier_eps)

        classifier.train()
        for ep in range(_new_classifier_eps):
            train_loss, correct, total = 0, 0, 0
            for batch_idx, (inputs, targets) in enumerate(new_data_loader):
                inputs, targets = inputs.float().to(device), targets.long().to(device)
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            scheduler.step()
            if ep % 25 == 0:
                logger.info("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{}), Temp List: {}".format(
                        ep,
                        batch_idx, len(trainloader), train_loss/(batch_idx+1), 
                        100.*correct/total, correct, total, temp_exps_list))

        test_loss, correct, total = 0, 0, 0
        assert args.dataset == "cifar100"
        client_sub_classes = {}
        for client_index in range(args.num_clients):
            sub_class_indices = [np.array_split(
                                    np.where(CIFAR100_COARSE_LABELS == sc_index)[0], 
                                    args.num_clients)[client_index].tolist() for sc_index in range(20)]
            client_sub_classes[client_index] = list(itertools.chain(*sub_class_indices))

        assert len(net_local_list) == args.num_clients
        with torch.no_grad():
            for batch_dix, (data, target) in enumerate(testloader):
                target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
                data = data.to(device)
                #new_test = np.zeros((args.var_bs, int(layer_idx * 20)))
                new_test = np.zeros((data.size()[0], int(layer_idx * 20)))
                client_shift_idx = 0
                for net_local_index, net_local in enumerate(net_local_list):
                    if net_local_index in temp_exps_list:
                        output = net_local(data)
                        smax = to_np(F.softmax(output, dim=1))
                        new_test[:, 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                        client_shift_idx+=1

                new_test = torch.from_numpy(new_test).float().to(device)
                output = classifier(new_test)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            logger.info("@@ Layer: {}, Tried Expert ID: {} Final {}/{}, Accuracy: {:.2f}%".format(
                    layer_idx, expert_index, correct, total, correct/total*100.0))
        final_accs.append(correct/total*100.0)
    logger.info("== Current Layer Acc List: {}, The Best Acc: {} ==".format(final_accs, np.max(final_accs)))
    return np.argmax(final_accs)


def oracle_classifier(net_local_list, trainloader, testloader, args, init_exps_list=None):
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100

    device = get_device()

    if not init_exps_list:
        cur_exps_list = []
        init_exps_list = []
        #total_layers = args.num_clients
    else:
        cur_exps_list = copy.deepcopy(init_exps_list)
        #total_layers = args.num_clients - len(init_exps_list)
    for layer_idx in range(len(init_exps_list)+1, args.num_clients+1):
        # we have num clients layers in total
        layer_selected_exp_idx = oracle_layer(net_local_list, trainloader, testloader, layer_idx, cur_exps_list, args, device)
        cur_exps_list.append(layer_selected_exp_idx)


def feature_constructor(net_local_list, trainloader, exps_list, args, device, train=False):
    # we only support label pred now:
    _classification_class = 20
    if train:
        _bs = args.local_bs
    else:
        _bs = args.var_bs
    # we only support label pred now:
    new_dataset = np.zeros((len(trainloader.dataset), int(_classification_class * len(exps_list))))
    new_labels = np.ones(len(trainloader.dataset))
    for batch_idx, data in enumerate(trainloader):
        with torch.no_grad():
            images, ori_labels = data
            labels = CIFAR100_COARSE_LABELS[ori_labels]
            images = images.to(device)
            # constructing datasets
            client_shift_idx = 0
            for client_i in range(args.num_clients):
                if client_i in exps_list:
                    net = net_local_list[client_i]
                    output = net(images)
                    smax = to_np(F.softmax(output, dim=1))
                    new_dataset[_bs*batch_idx:_bs*(batch_idx+1), 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                    client_shift_idx += 1
        new_labels[_bs*batch_idx:_bs*(batch_idx+1)] = labels
    return new_dataset, new_labels


def tree_oracle_classifier(net_local_list, testloader, val_indices, args):
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100
    mean = [0.5071, 0.4867, 0.4408]
    std  = [0.2675, 0.2565, 0.2761]
    transform_train = transforms.Compose([
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
    dummy_dataset_train = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
    dummy_dataset_test = datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
    dataset_val = copy.deepcopy(dummy_dataset_train)
    dataset_val.data = dataset_val.data[val_indices]
    dataset_val.targets = np.array(dataset_val.targets)[val_indices]
    trainloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.local_bs, 
                                                                shuffle=True,
                                                                num_workers=4,
                                                                pin_memory=True)
    device = get_device()

    num_clusters = 5
    exps_list = [0,19]
    new_dataset_train, _ = feature_constructor(net_local_list, trainloader, exps_list, args, device, train=True)
    new_dataset_test, _ = feature_constructor(net_local_list, testloader, exps_list, args, device)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(new_dataset_train)
    test_clusters = kmeans.predict(new_dataset_test)

    for i in range(num_clusters):
        indices_train = np.where(kmeans.labels_ == i)[0]
        sliced_dataset_train = copy.deepcopy(dummy_dataset_train)
        sliced_dataset_train.data = sliced_dataset_train.data[val_indices][indices_train]
        sliced_dataset_train.targets = np.array(sliced_dataset_train.targets)[val_indices][indices_train]

        indices_test = np.where(test_clusters == i)[0]
        sliced_dataset_test = copy.deepcopy(dummy_dataset_test)
        sliced_dataset_test.data = sliced_dataset_test.data[indices_test]
        sliced_dataset_test.targets = np.array(sliced_dataset_test.targets)[indices_test]
        print("!!! sliced train: {}, sliced test: {}".format(indices_train, indices_test))

        print("!!! cluster: {}, sliced train data size: {}, sliced test data size: {}!!!".format(i, sliced_dataset_train.data.shape,
                                sliced_dataset_test.data.shape))
        cluster_train_loader = torch.utils.data.DataLoader(sliced_dataset_train, batch_size=args.local_bs, 
                                                                shuffle=True,
                                                                num_workers=4,
                                                                pin_memory=True)
        cluster_test_loader = torch.utils.data.DataLoader(sliced_dataset_test, batch_size=args.var_bs, 
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True)
        oracle_classifier(
                net_local_list, cluster_train_loader, cluster_test_loader, args, 
                init_exps_list=exps_list
            )


def classifier_trainer(net_local_list, trainloader, valoader, testloader, sampled_expert_list, args, run_test=True):
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100
    device = get_device()
    _classification_class = 20

    new_dataset = np.zeros((len(trainloader.dataset), 
                        int(_classification_class * len(sampled_expert_list))))
    new_labels = np.zeros(len(trainloader.dataset))
    print("** sampled experts: {}".format(sampled_expert_list))

    for batch_idx, data in enumerate(trainloader):
        with torch.no_grad():
            images, ori_labels = data
            labels = CIFAR100_COARSE_LABELS[ori_labels]
            images = images.to(device)
            client_shift_idx = 0
            for client_i in range(args.num_clients):
                if client_i in sampled_expert_list:
                    net = net_local_list[client_i]
                    output = net(images)
                    smax = to_np(F.softmax(output, dim=1))
                    new_dataset[args.local_bs*batch_idx:args.local_bs*(batch_idx+1), 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                    new_labels[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = labels
                    client_shift_idx += 1

    if args.sp_model == "nn":
        classifier = SimpleClassifier(num_clients=len(sampled_expert_list), num_classes=_classification_class).to(device)
        _new_classifier_eps = 61
        new_data_loader = torch.utils.data.DataLoader(SimpleDataset(data=new_dataset, targets=new_labels),
                                                        batch_size=128, shuffle=True, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_classifier_eps)

        classifier.train()
        for ep in range(_new_classifier_eps):
            train_loss, correct, total = 0, 0, 0
            for batch_idx, (inputs, targets) in enumerate(new_data_loader):
                inputs, targets = inputs.float().to(device), targets.long().to(device)
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            if ep % 30 == 0:
                logger.info("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                            ep,
                            batch_idx, len(trainloader), train_loss/(batch_idx+1), 
                            100.*correct/total, correct, total))
            scheduler.step()
    elif args.sp_model == "knn":
        classifier = KNeighborsClassifier(n_neighbors=9)
        classifier.fit(new_dataset, new_labels)
        logger.info("Done `fitting` the kNN classifier ...")
    else:
        raise NotImplementedError("Unsupported SP model type ...")

    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        if args.sp_model == "nn":
            classifier.eval()
        for batch_dix, (data, target) in enumerate(valoader):
            target = torch.from_numpy(np.array(CIFAR100_COARSE_LABELS[target])).to(device)
            data = data.to(device)
            new_test = np.zeros((data.size()[0], int(len(sampled_expert_list) * _classification_class)))
            client_shift_idx = 0
            for net_local_index, net_local in enumerate(net_local_list):
                if net_local_index in sampled_expert_list:
                    output = net_local(data)
                    smax = to_np(F.softmax(output, dim=1))
                    new_test[:, 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                    client_shift_idx+=1
            if args.sp_model == "nn":
                new_test = torch.from_numpy(new_test).float().to(device)
                output = classifier(new_test)
                pred = output.argmax(dim=1, keepdim=True)
            elif args.sp_model == "knn":
                pred = torch.from_numpy(classifier.predict(new_test)).to(device)
            else:
                raise NotImplementedError("Unsupported SP model type ...")
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += pred.size(0)
        fin_val_acc = val_correct/val_total*100.0

    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        if args.sp_model == "nn":
            classifier.eval()
        for batch_dix, (data, target) in enumerate(testloader):
            target = torch.from_numpy(np.array(CIFAR100_COARSE_LABELS[target])).to(device)
            data = data.to(device)
            new_test = np.zeros((data.size()[0], int(len(sampled_expert_list) * _classification_class)))
            client_shift_idx = 0
            for net_local_index, net_local in enumerate(net_local_list):
                if net_local_index in sampled_expert_list:
                    output = net_local(data)
                    smax = to_np(F.softmax(output, dim=1))
                    new_test[:, 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                    client_shift_idx+=1
            if args.sp_model == "nn":
                new_test = torch.from_numpy(new_test).float().to(device)
                output = classifier(new_test)
                pred = output.argmax(dim=1, keepdim=True)
            elif args.sp_model == "knn":
                pred = torch.from_numpy(classifier.predict(new_test)).to(device)
            else:
                raise NotImplementedError("Unsupported SP model type ...")
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += pred.size(0)
        fin_test_acc = test_correct/test_total*100.0
        logger.info("@@ Final {}/{}, Val Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(
                test_correct, test_total, fin_val_acc, fin_test_acc))
    return classifier, fin_val_acc, fin_test_acc

def classifier_generator(net_local_list, trainloader, valoader, testloader, args, num_apis=20, 
                            resume=False):
    _file_name = 'trained_apis_{}apis_{}.pt'.format(num_apis, args.sp_model)

    def _regular_classifer_generation(net_local_list, trainloader, valoader, testloader, args, num_apis):
        fist_layer_choice = []
        classifier_collector = {} # key: (tuple) API pair; value: classifier
        for n_api in range(1, num_apis+1):
            if args.sp_model == "knn":
                # for kNN, let's fit the model on the fly
                # for here, we only aim at testing some evaluation results
                trial_list = [1] + [i for i in range(2, num_apis+1, 2)]
                if n_api not in trial_list:
                    continue
            elif args.sp_model == "nn":
                if n_api > 5:
                    break
            else:
                raise NotImplementedError("Unsupported SP model type ...")
            if n_api == 1:
                acc_collector = []
            n_api_subset = findsubsets(range(num_apis), n_api)
            if args.sp_model == "knn":
                _num_elems_to_sample = 100
                if len(n_api_subset) > _num_elems_to_sample:
                    n_api_subset = random.sample(n_api_subset, _num_elems_to_sample)
            for tp_idx, tp in enumerate(n_api_subset):
                if not set(fist_layer_choice).issubset(set(tp)):
                    continue
                logger.info("@ Training for Comb: {}, fist_layer_choice: {}".format(tp, fist_layer_choice))
                trained_classifier, fin_val_acc, fin_test_acc = classifier_trainer(
                        net_local_list, trainloader, valoader, testloader, tp, args
                    )
                classifier_collector[tp] = {'model':trained_classifier,
                                            'test_acc':fin_test_acc}
                if n_api == 1:
                    acc_collector.append(fin_val_acc)
            if n_api == 1:
                fist_layer_choice.append(np.argmax(acc_collector))
        classifier_collector['fist_layer_choice'] = fist_layer_choice
        return classifier_collector

    def _beam_search_classifer_generation(net_local_list, trainloader, valoader, testloader, args, num_apis):
        fist_layer_choice, api_comb_choices, layer_choices_col = [], [], {}
        classifier_collector = {} # key: (tuple) API pair; value: classifier
        for n_api in range(1, num_apis+1):
            acc_collector, tp_collector = [], []
            n_api_subset = findsubsets(range(num_apis), n_api)
            for tp_idx, tp in enumerate(n_api_subset):
                if n_api > 1:
                    if not any([set(api_choice).issubset(set(tp)) for api_choice in api_comb_choices]):
                        continue
                logger.info("@ Training for Comb: {}, fist_layer_choice: {}, API choices: {}".format(
                                        tp, fist_layer_choice, api_comb_choices))
                trained_classifier, fin_val_acc, fin_test_acc = classifier_trainer(
                        net_local_list, trainloader, valoader, testloader, tp, args
                    )
                classifier_collector[tp] = {'model':trained_classifier,
                                            'test_acc':fin_test_acc}
                acc_collector.append(fin_val_acc)
                tp_collector.append(tp)
            if n_api == 1:
                max_index = np.argmax(acc_collector)
                fist_layer_choice.append(max_index)
                api_comb_choices = [tp_collector[max_index]]
            else:
                layer_indices = topk_indices(acc_collector, k=args.num_apis_per_layer_bs)
                api_comb_choices = np.array(tp_collector)[layer_indices].tolist()

            layer_choices_col[n_api] = api_comb_choices

        classifier_collector['fist_layer_choice'] = fist_layer_choice
        classifier_collector['layer_choices_collections'] = layer_choices_col
        return classifier_collector

    if not resume:
        if args.beam_search:
            classifier_collector = _beam_search_classifer_generation(
                                        net_local_list, trainloader, valoader, testloader, args, num_apis
                                    )
        else:
            classifier_collector = _regular_classifer_generation(
                                        net_local_list, trainloader, valoader, testloader, args, num_apis
                                    )
        with open(_file_name, 'wb') as f:
            torch.save(classifier_collector, f)
    else:
        with open(_file_name, 'rb') as f:
            classifier_collector = torch.load(f)
    return classifier_collector


def test_feature_constructor(net_local_list, test_sample, exps_list, args, device):
    # we only support label pred now:
    _classification_class = 20

    # we only support label pred now:
    new_dataset = np.zeros(int(_classification_class * len(exps_list)))
    
    with torch.no_grad():
        client_shift_idx = 0
        for client_i in range(args.num_clients):
            if client_i in exps_list:
                net = net_local_list[client_i]
                output = net(test_sample)
                smax = to_np(F.softmax(output, dim=1))
                new_dataset[20*client_shift_idx:20*(client_shift_idx+1)] = smax
                client_shift_idx += 1
    return new_dataset


def shortest_path_loaders(val_indices, testloader, args):
    # we further split the validation set into "shortest path train|shortest path val"
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    val_indices, len_val_idxs = np.random.permutation(val_indices), len(val_indices)
    if args.sp_model == "nn":
        sp_train_idxs = val_indices[:int(0.9*len_val_idxs)]
        sp_val_idxs = val_indices[int(0.9*len_val_idxs):]
    elif args.sp_model == "knn":
        sp_train_idxs = val_indices[:int(0.99*len_val_idxs)]
        sp_val_idxs = val_indices[int(0.99*len_val_idxs):]
    else:
        raise NotImplementedError("Unsupported model type ...")
    transform_train = transforms.Compose([
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
    dummy_dataset_train = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
    dummy_dataset_val = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_test)
    dataset_train, dataset_val = copy.deepcopy(dummy_dataset_train), copy.deepcopy(dummy_dataset_val)
    # train ds for sp
    dataset_train.data = dataset_train.data[sp_train_idxs]
    dataset_train.targets = np.array(dataset_train.targets)[sp_train_idxs]
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.local_bs, 
                                                                shuffle=True,
                                                                num_workers=4,
                                                                pin_memory=True)
    # val ds for sp
    dataset_val.data = dataset_val.data[sp_val_idxs]
    dataset_val.targets = np.array(dataset_val.targets)[sp_val_idxs]
    valoader = torch.utils.data.DataLoader(dataset_val, batch_size=args.local_bs, 
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True)

    # test loader for benchmarking
    testloader_bechmark = torch.utils.data.DataLoader(testloader.dataset,
                                                        batch_size=500, 
                                                        shuffle=False,
                                                        num_workers=4,
                                                        pin_memory=True)
    del dummy_dataset_train
    del dummy_dataset_val
    return trainloader, valoader, testloader_bechmark


def shortest_path_approach(net_local_list, val_indices, testloader, args):
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100
    _NUM_APIS = 20
    _K4KNN = args.k4knn
    _DELTA_DIST_THRESHOLD = args.delta_dist_threshold
    device = get_device()

    def _get_dist(nearest_indices, api_dataset_val, new_label_val, api_to_query, args):
        total, correct = 0, 0
        nearest_api_target = torch.from_numpy(
                                       new_label_val[nearest_indices]).to(device)
        if args.sp_model == "nn":
            nearest_api_data = torch.from_numpy(
                                       api_dataset_val[nearest_indices, :]).float().to(device)
            output = api_to_query(nearest_api_data)
            pred = output.argmax(dim=1, keepdim=True)
        elif args.sp_model == "knn":
            pred = torch.from_numpy(api_to_query.predict(api_dataset_val[nearest_indices, :])).to(device)
        else:
            raise NotImplementedError("Unsupported SP model ...")
        acc = pred.eq(nearest_api_target.view_as(pred)).sum().item() / pred.size(0)
        dist = 100.0 - (acc * 100.0)
        return dist

    def _get_classifier(classifier_collector, ds, labels, key, args):
        # for NN, we just fetch the correct classifier
        if args.sp_model == "nn":
            classifier = classifier_collector[key]['model']
        # for kNN, we fit a classifier on the fly 
        elif args.sp_model == "knn":
            classifier = KNeighborsClassifier(n_neighbors=9)
            classifier.fit(ds, labels)
        else:
            raise NotImplementedError("Unsupported SP model ...")
        return classifier

    trainloader, valoader, testloader_bechmark = shortest_path_loaders(val_indices, testloader, args)
    exps_list = [i for i in range(args.num_clients)]
    new_dataset_val, new_label_val = feature_constructor(
                                            net_local_list, 
                                            valoader, 
                                            exps_list, args, device, 
                                            train=True
                                        )
    new_dataset_train, new_label_train = feature_constructor(
                                            net_local_list, 
                                            trainloader, 
                                            exps_list, args, device, 
                                            train=True
                                        )

    # generate all possible combinations:
    classifier_collector = classifier_generator(net_local_list, trainloader, valoader, testloader_bechmark, 
                                args, 
                                num_apis=args.num_clients,
                                resume=True)
    oracle_accs, random_accs = {}, {}
    for i in range(1, args.num_clients+1):
        oracle_accs[i], random_accs[i] = [], 0.0
    for k, v in classifier_collector.items():
        if isinstance(v, dict):
            if 'test_acc' in v.keys():
                logger.info("*** key: {}, value: {}".format(k, v['test_acc']))
                oracle_accs[len(k)].append(v['test_acc'])
            elif k == "layer_choices_collections":
                for kk, vv in v.items():
                    logger.info("layer: {}, choices: {}".format(kk, vv
                                                            ))
    layers_rand_accs, layers_orcle_accs = [], []
    for k, v in oracle_accs.items():
        if len(v) != 0:
            layers_rand_accs.append(np.random.choice(v, 1, replace=False)[0])
            layers_orcle_accs.append(max(v))
    logger.info("*** orcale accs : {}, random accs: {}".format(layers_orcle_accs,
                                                            layers_rand_accs))

    test_correct, test_total, avg_num_apis = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            test_target = torch.from_numpy(CIFAR100_COARSE_LABELS[np.array(target)]).to(device)
            test_data = data.to(device)

            visited_node_list = copy.deepcopy(classifier_collector['fist_layer_choice'])
            if args.beam_search:
                actual_visited_nodes = copy.deepcopy(classifier_collector['fist_layer_choice'])
            current_min_dist = float('inf')

            if args.sp_model == "nn":
                _NUM_APIS = 5
            for n_api in range(2, _NUM_APIS+1):
                logger.info("* visited_node_list: {}".format(visited_node_list))
                api_data_test = test_feature_constructor(net_local_list, test_data, 
                                                    visited_node_list, args, device)
                api_dataset_val4knn = np.concatenate(
                                                [new_dataset_val[:, 20*tt:20*(tt+1)] for tt in np.sort(visited_node_list)], 
                                                axis=1)
                api_dataset_train4knn = np.concatenate(
                                                [new_dataset_train[:, 20*tt:20*(tt+1)] for tt in np.sort(visited_node_list)], 
                                                axis=1)

                # find knn
                if args.sp_model == "nn":
                    nearest_indices = find_nearest_indices(api_data_test, api_dataset_val4knn, N=_K4KNN)
                elif args.sp_model == "knn":
                    nearest_indices = find_nearest_indices(api_data_test, api_dataset_train4knn, N=_K4KNN)
                else:
                    raise NotImplementedError("...")

                if n_api > 2:
                    api_last_round = _get_classifier(classifier_collector, 
                                                    api_dataset_train4knn, new_label_train, 
                                                    key=min_dist_tp, args=args
                                                )
                    # update the cur min dist
                    if args.sp_model == "nn":
                        current_min_dist = _get_dist(
                                            nearest_indices, api_dataset_val4knn, new_label_val, api_last_round, args
                                        )
                    elif args.sp_model == "knn":
                        current_min_dist = _get_dist(
                                            nearest_indices, api_dataset_train4knn, new_label_train, api_last_round, args
                                        )
                    else:
                        raise NotImplementedError("...")
                    logger.info("@@@ update API list: {}, updated dist: {}".format(min_dist_tp, current_min_dist))

                # in each layer we select the one API that leads to the smallest pred err (dist)
                if args.beam_search:
                    n_api_subset = []
                    for node_comb in classifier_collector.keys():
                        if isinstance(node_comb, tuple) and len(node_comb) == n_api:
                            n_api_subset.append(node_comb)
                else:
                    n_api_subset = findsubsets(range(args.num_clients), n_api)

                tempt_dist, tempt_tp = [], []
                for tp in n_api_subset:
                    if not args.beam_search:
                        if not set(visited_node_list).issubset(set(tp)):
                            continue
                    else:
                        tp = tuple(tp)

                    api_dataset_val = np.concatenate(
                                                [new_dataset_val[:, 20*tt:20*(tt+1)] for tt in np.sort(tp)], 
                                                axis=1)
                    api_dataset_train = np.concatenate(
                                                [new_dataset_train[:, 20*tt:20*(tt+1)] for tt in np.sort(tp)], 
                                                axis=1)
                    api_to_query = _get_classifier(classifier_collector, 
                                                    api_dataset_train, new_label_train, 
                                                    key=tp, args=args
                                                )
                    if args.sp_model == "nn":
                        dist = _get_dist(nearest_indices, api_dataset_val, new_label_val, api_to_query, args)
                    elif args.sp_model == "knn":
                        dist = _get_dist(nearest_indices, api_dataset_train, new_label_train, api_to_query, args)
                    else:
                        raise NotImplementedError("...")

                    tempt_dist.append(dist)
                    tempt_tp.append(tp)
                    logger.info("* tp: {}, dist: {:.4f}".format(tp, dist))
                min_dist_idx = np.argmin(tempt_dist)
                _delta_dist = current_min_dist-tempt_dist[min_dist_idx]
                logger.info("@@@ delta min dist: {}".format(_delta_dist))
                if _delta_dist < _DELTA_DIST_THRESHOLD:
                    break
                min_dist_tp, current_min_dist = tempt_tp[min_dist_idx], tempt_dist[min_dist_idx]
                
                diff_elem = list(set(min_dist_tp) - set(visited_node_list))
                visited_node_list.append(diff_elem[0])
                if args.beam_search:
                    diff_elems = list(set(min_dist_tp) - set(actual_visited_nodes))
                    actual_visited_nodes.extend(diff_elems)

            logger.info("*** fin visited_node_list: {}, min_dist_tp: {}".format(
                                            visited_node_list, min_dist_tp))
            fin_api_test = test_feature_constructor(net_local_list, 
                                test_data, visited_node_list, args, device)
            api_dataset_train4test = np.concatenate(
                                        [new_dataset_train[:, 20*tt:20*(tt+1)] for tt in np.sort(visited_node_list)], 
                                        axis=1
                                    )
            api_test = _get_classifier(classifier_collector, 
                                        api_dataset_train4test, new_label_train, 
                                        key=min_dist_tp, args=args
                                    )
            test_target = test_target.to(device)
            if args.sp_model == "nn":
                fin_api_test = torch.from_numpy(fin_api_test).view(1, -1).float().to(device)
                output_test = api_test(fin_api_test)
                pred_test = output_test.argmax(dim=1, keepdim=True)
            elif args.sp_model == "knn":
                fin_api_test = fin_api_test.reshape(1, -1)
                pred_test = torch.from_numpy(api_test.predict(fin_api_test)).to(device)
            else:
                raise NotImplementedError("Unsupported SP model ...")
            test_correct += pred_test.eq(test_target.view_as(pred_test)).sum().item()
            test_total += 1

            if args.beam_search:
                avg_num_apis += len(actual_visited_nodes)
            else:
                avg_num_apis += len(min_dist_tp)
            logger.info("!!!!!! test_correct: {}, test_total: {}, avg_num_apis: {} !!!!!!".format(
                            test_correct, test_total, avg_num_apis/test_total))


def frugal_ml_logger(net_local_list, trainloader, args):
    if args.use_histgram:
        cross_expert_confs = []
    else:
        cross_expert_confs = np.empty((args.num_clients, args.num_clients))
    assert args.dataset == "cifar100" # currently we only support corss expert acc for cifar-100

    device = get_device()
    classifier = SimpleClassifier(num_clients=args.num_clients, num_classes=20).to(device)


    #new_dataset = np.zeros((len(trainloader.dataset), int(args.num_clients*20)))
    #new_labels = np.zeros(len(trainloader.dataset))
    confidence = np.zeros(len(trainloader.dataset))
    predictedlabels = np.zeros(len(trainloader.dataset))
    truelabels = np.zeros(len(trainloader.dataset))
    reward = np.zeros(len(trainloader.dataset))

    # constructing datasets
    for client_i in range(args.num_clients):
        based_name = "./frugalml_gen_data/Model{}_".format(client_i)

        for batch_idx, data in enumerate(trainloader):
            with torch.no_grad():
                images, labels = data
                labels = CIFAR100_COARSE_LABELS[labels]
                pt_labels = torch.from_numpy(labels).to(device)
                images = images.to(device)

                net = net_local_list[client_i]
                output = net(images)
                smax = to_np(F.softmax(output, dim=1))
                scores = np.max(smax, axis=1)
                confidence[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = scores
                truelabels[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = labels.astype(np.int32)
                pred = output.argmax(dim=1, keepdim=True)
                predictedlabels[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = to_np(pred).flatten().astype(np.int32)
                reward[args.local_bs*batch_idx:args.local_bs*(batch_idx+1)] = to_np(pred.eq(pt_labels.view_as(pred))).flatten()

        accm_reward = np.zeros(len(trainloader.dataset))
        accum_reward_elem = 0
        for i, ar in enumerate(reward):
            accum_reward_elem += ar
            accm_reward[i] = accum_reward_elem

        with open(based_name+"Confidence.txt", 'w') as file:
            for element in confidence:
                file.write('{:.2f}\n'.format(element))
        with open(based_name+"ImageName.txt", 'w') as file:
            for elem_id, element in enumerate(confidence):
                file.write('validation_{}.jpg\n'.format(elem_id))
        with open(based_name+"PredictedLabel.txt", 'w') as file:
            for elem_id, element in enumerate(predictedlabels):
                file.write('{:d}\n'.format(int(element)))
        with open(based_name+"Reward.txt", 'w') as file:
            for elem_id, element in enumerate(reward):
                file.write('{:d}\n'.format(int(element)))
        with open(based_name+"TotalReward.txt", 'w') as file:
            for elem_id, element in enumerate(accm_reward):
                file.write('{:d}\n'.format(int(element)))
        with open(based_name+"TrueLabel.txt", 'w') as file:
            for elem_id, element in enumerate(truelabels):
                file.write('{:d}\n'.format(int(element)))

def global_inference(net_local_list, test_loader, args):
    # the most vanilla implementation
    test_loss = 0
    correct = 0
    total = 0

    device = get_device()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            net_var_list = np.zeros(args.num_clients)
            for net_local_index, net_local in enumerate(net_local_list):
                pred_list = np.zeros(args.var_iter_budget)
                for t in range(args.var_iter_budget):
                    net_local.train()
                    output = net_local(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    pred_list[t] = pred.data
                batch_var = np.var(pred_list)
                net_var_list[net_local_index] = batch_var
            min_var_idx = np.argmin(net_var_list)
            net_local_list[min_var_idx].eval()
            output = net_local_list[min_var_idx](data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            print("* Correct Pred: {}, Total: {}, Selected expert: {}, True label: {}".format(correct, total, min_var_idx, target.item()))
        print("@@ Final {}/{}, Accuracy: {:.2f}%".format(correct, total, correct/total*100.0))


def test_local(net, testloader, no_print=False, client_idx=0, client_class_indices=None, args=None):
    device = get_device()
    if args.dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == "cifar100":
        classes = ('aquatic-mammals', 'fish', 'flowers', 'food-containers', 'fruit&vegetables',
                    'household-E-devices', 'household-furniture', 'insects', 'L-carnivores', 'L-man-made-outdoor-things',
                    'L-natural-outdoor-scenes', 'L-omnivores&herbivores', 'medium-sized-mammals', 'non-insect-invertebrates', 'people',
                    'reptiles', 'small-mammals', 'trees', 'vehicles1', 'vehicles2'
                    )
        # the i-th user gets the data points in the i-th subclass of all super classes
        if args.par_strategy == "corse-label":
            sub_class_indices = [np.array_split(np.where(CIFAR100_COARSE_LABELS == sc_index)[0], args.num_clients)[client_idx].tolist() for sc_index in range(20)]
            client_sub_classes = list(itertools.chain(*sub_class_indices))
            client_sub_labels = [CIFAR100_SUBCLASSES[sc_index][client_idx] for sc_index in range(20)]
        elif args.par_strategy == "sample-based":
            client_sub_classes = client_class_indices
            client_sub_labels = [CIFAR100_FINE_LABELS[int(cci)] for cci in client_class_indices]            
        else:
            raise NotImplementedError("Unsupported par strategy ...")
    else:
        raise NotImplementedError("Unsupported dataset ...")
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    preds_sm = []
    labels_oneh = []

    acc_stat = {
        "correct" : 0,
        "total" : 0,
        "id_correct" : 0,
        "id_total" : 0,
        "ood_correct" : 0,
        "ood_total" : 0,
    }
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if args.dataset == "cifar100":
                super_labels = torch.from_numpy(CIFAR100_COARSE_LABELS[labels]).to(device)
                L = super_labels
            elif args.dataset == "cifar10":
                L = labels
            else:
                raise NotImplementedError("Unsupported dataset ...")
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            pred_sm = outputs.cpu().detach()

            
            if args.dataset == "cifar100":
                acc_stat["total"] += super_labels.size(0)
                acc_stat["correct"] += (predicted == super_labels).sum().item()
            else:
                acc_stat["total"] += labels.size(0)
                acc_stat["correct"] += (predicted == labels).sum().item()

            # Convert labels to one hot encoding
            label_oneh = labels.cpu().detach()

            preds_sm.append(pred_sm)
            labels_oneh.append(label_oneh)

            # collect the correct predictions for each class
            for label, ll, prediction in zip(L, labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

                if args.dataset == "cifar100":
                    if ll.item() in client_sub_classes:
                        if label == prediction:
                            acc_stat["id_correct"] += 1
                        acc_stat["id_total"] += 1
                    else:
                        if label == prediction:
                            acc_stat["ood_correct"] += 1
                        acc_stat["ood_total"] += 1
                
    if not no_print:
        # print accuracy for each class
        for (classname, correct_count), csl in zip(correct_pred.items(), client_sub_labels):
            accuracy = 100 * float(correct_count) / total_pred[classname]
            logger.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %, InDist Label: {csl:5s}')

        if args.dataset == "cifar10":
            logger.info("Dataset: {}; Local Valid. Acc.: {:.2f}".format(
                    args.dataset, 100 * acc_stat["correct"] / acc_stat["total"]))
        elif args.dataset == "cifar100":
            logger.info("Dataset: {}; Local Valid. Acc.: {:.2f}, InDist. Acc. : {:.2f}, OOD. Acc. {:.2f}".format(
                    args.dataset, 100 * acc_stat["correct"] / acc_stat["total"],
                    100 * acc_stat["id_correct"] / acc_stat["id_total"],
                    100 * acc_stat["ood_correct"] / acc_stat["ood_total"]
                    ))            
    preds_sm = torch.cat(preds_sm, dim=0)
    labels_oneh = torch.cat(labels_oneh, dim=0)

    return preds_sm, labels_oneh


def global_inference_score_based(net_local_list, test_loader, args, client_class_indices, score="energy"):
    test_loss = 0
    correct = 0
    total = 0
    device = get_device()

    if args.dataset == "cifar100":
        client_sub_classes = {}
        for client_index in range(args.num_clients):
            # the i-th user gets the data points in the i-th subclass of all super classes
            #sub_class_indices = [np.where(CIFAR100_COARSE_LABELS == sc_index)[0][client_index] for sc_index in range(20)]
            #client_sub_classes[client_index] = sub_class_indices
            sub_class_indices = [np.array_split(
                                    np.where(CIFAR100_COARSE_LABELS == sc_index)[0], 
                                    args.num_clients)[client_index].tolist() for sc_index in range(20)]
            client_sub_classes[client_index] = list(itertools.chain(*sub_class_indices))

    with torch.no_grad():
        expert_assignment_list = []
        for batch_dix, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            net_var_list = np.zeros((args.var_bs, args.num_clients))
            for net_local_index, net_local in enumerate(net_local_list):
                output = net_local(data)
                if score == "softmax":
                    smax = to_np(F.softmax(output, dim=1))
                    scores = np.max(smax, axis=1)
                elif score == "energy":
                    scores = to_np((args.T*torch.logsumexp(output / args.T, dim=1)))
                else:
                    raise NotImplementedError("Unsupported score type ...")
                net_var_list[:, net_local_index] = scores

            expert_assignment = np.argmax(net_var_list, axis=1)
            expert_assignment_list.extend(expert_assignment.tolist())

        # queuing inference into batches, and wire to appropriate expert network
        digit_queues = [[] for _ in range(args.num_clients)]
        label_queues = [[] for _ in range(args.num_clients)]

        perfect_digit_queues = [[] for _ in range(args.num_clients)]
        perfect_label_queues = [[] for _ in range(args.num_clients)] 

        image_idx_counter = 0
        expert_assign_correct = 0
        expert_assign_total = 0
        for data, target in test_loader:
            if args.dataset == "cifar100":
                ori_target = target
                target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
            for image_idx in range(args.var_bs):
                image_exprt_assignment = expert_assignment_list[image_idx_counter]
                digit_queues[image_exprt_assignment].append(data[image_idx, :, :, :])
                label_queues[image_exprt_assignment].append(target[image_idx].data)

                if ori_target[image_idx] in client_class_indices[image_exprt_assignment, :].tolist():
                    expert_assign_correct += 1
                expert_assign_total += 1
                image_idx_counter += 1

                correct_expert_assignment = []
                for c_index in range(args.num_clients):
                    if ori_target[image_idx] in client_class_indices[c_index, :].tolist():
                        correct_expert_assignment.append(c_index)
                
                for cea in correct_expert_assignment:
                    perfect_digit_queues[cea].append(data[image_idx, :, :, :])
                    perfect_label_queues[cea].append(target[image_idx].data)
        logger.info("** Expert Selection Accuracy: {}/{}, {:.2f}%".format(expert_assign_correct, expert_assign_total, 
                                                        expert_assign_correct/expert_assign_total*100.0))

        expert_indices = []
        correct_labels = []
        for expert_idx, (digits, labels) in enumerate(zip(digit_queues, label_queues)):
            reconstructed_data = torch.stack(digits, dim=0).to(device)
            reconstructed_targets = torch.stack(labels, dim=0).to(device)
            net_local_list[expert_idx].eval()
            output = net_local_list[expert_idx](reconstructed_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(reconstructed_targets.view_as(pred)).sum().item()
            total += reconstructed_targets.size(0)

            # expert_indices.extend([expert_idx for _ in labels])
            # if args.dataset == "cifar10":
            #     correct_labels.extend([l.item() for l in labels])
            # elif args.dataset == "cifar100":
            #     tempt_cls = []
            #     for l in labels:
            #         for item_id, (k, v) in enumerate(client_sub_classes.items()):
            #             if l in v:
            #                 tempt_cls.append(k)
            #     correct_labels.extend(tempt_cls)
            # else:
            #     raise NotImplementedError("Unsupported Dataset ...")

        perfect_correct = 0
        perfect_total = 0
        for expert_idx, (digits, labels) in enumerate(zip(perfect_digit_queues, perfect_label_queues)):
            reconstructed_data = torch.stack(digits, dim=0).to(device)
            reconstructed_targets = torch.stack(labels, dim=0).to(device)
            net_local_list[expert_idx].eval()
            output = net_local_list[expert_idx](reconstructed_data)
            pred = output.argmax(dim=1, keepdim=True)
            perfect_correct += pred.eq(reconstructed_targets.view_as(pred)).sum().item()
            perfect_total += reconstructed_targets.size(0)

            # if args.dataset == "cifar10":
            #     correct_labels.extend([l.item() for l in labels])
            # elif args.dataset == "cifar100":
            #     tempt_cls = []
            #     for l in labels:
            #         for item_id, (k, v) in enumerate(client_sub_classes.items()):
            #             if l in v:
            #                 tempt_cls.append(k)
            #     correct_labels.extend(tempt_cls)
            # else:
            #     raise NotImplementedError("Unsupported Dataset ...")

        #conf_mat = confusion_matrix(y_true=expert_indices, y_pred=correct_labels)
        #logger.info("@@ Final {}/{}, Accuracy: {:.2f}%, Perfect Acc.: {:.2f}%,, conf. mat.: \n{}\n".format(
        #        correct, total, correct/total*100.0, perfect_correct/perfect_total*100.0, conf_mat))
        logger.info("@@ Final {}/{}, Accuracy: {:.2f}%, Perfect Acc.: {:.2f}%".format(
                correct, total, correct/total*100.0, perfect_correct/perfect_total*100.0))

def global_inference_mutual_conf(net_local_list, test_loader, args, client_class_indices, cross_client_info):
    test_loss = 0
    correct = 0
    total = 0
    device = get_device()

    if args.dataset == "cifar100":
        client_sub_classes = {}
        for client_index in range(args.num_clients):
            sub_class_indices = [np.array_split(
                                    np.where(CIFAR100_COARSE_LABELS == sc_index)[0], 
                                    args.num_clients)[client_index].tolist() for sc_index in range(20)]
            client_sub_classes[client_index] = list(itertools.chain(*sub_class_indices))

    with torch.no_grad():
        expert_assignment_list = []
        for batch_dix, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            net_var_list = np.zeros((args.var_bs, args.num_clients))
            for net_local_index, net_local in enumerate(net_local_list):
                output = net_local(data)
                smax = to_np(F.softmax(output, dim=1))

                if args.e_mat_score == "maxsoftmax":
                    scores = np.max(smax, axis=1)
                elif args.e_mat_score == "entropy":
                    scores = entropy(smax, axis=1)
                else:
                    raise NotImplementedError("Unsupported E mat Score ...")
                
                net_var_list[:, net_local_index] = scores # bsize x num clients
                cross_client_cos = np.zeros((args.var_bs, args.num_clients))

                if args.use_histgram:
                    for batch_idx in range(args.var_bs):
                        for client_i in range(args.num_clients):
                            temp_prob = np.zeros(args.num_clients)
                            for client_j in range(args.num_clients):
                                sub_histgram = cross_client_info[client_i][client_j]
                                for edge_idx, edge_val in enumerate(sub_histgram[1]):
                                    if edge_idx == 0:
                                        continue
                                    if net_var_list[batch_idx, client_i] <= edge_val:
                                        temp_prob[client_j] = sub_histgram[0][edge_idx-1]
                                        break
                            cross_client_cos[batch_idx, client_i] = temp_prob.sum()
                    expert_assignment = np.argmax(cross_client_cos, axis=1)    
                else:
                    for i in range(args.var_bs):
                        for j in range(args.num_clients):
                            if args.dist_metric == "cosine":
                                cross_client_cos[i][j] = cosine(
                                                        net_var_list[i, :], cross_client_info[j, :]
                                                    )
                            elif args.dist_metric == "l2":
                                cross_client_cos[i][j] = euclidean(
                                                        net_var_list[i, :], cross_client_info[j, :]
                                                    )
                            else:
                                raise NotImplementedError("Unsupported dist metric .")

                    expert_assignment = np.argmin(cross_client_cos, axis=1)
                expert_assignment_list.extend(expert_assignment.tolist())

        # queuing inference into batches, and wire to appropriate expert network
        digit_queues = [[] for _ in range(args.num_clients)]
        label_queues = [[] for _ in range(args.num_clients)]

        perfect_digit_queues = [[] for _ in range(args.num_clients)]
        perfect_label_queues = [[] for _ in range(args.num_clients)]        

        image_idx_counter = 0
        expert_assign_correct = 0
        expert_assign_total = 0
        for data, target in test_loader:
            if args.dataset == "cifar100":
                ori_target = target
                target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
            for image_idx in range(args.var_bs):
                image_exprt_assignment = expert_assignment_list[image_idx_counter]
                digit_queues[image_exprt_assignment].append(data[image_idx, :, :, :])
                label_queues[image_exprt_assignment].append(target[image_idx].data)

                if ori_target[image_idx] in client_class_indices[image_exprt_assignment, :].tolist():
                    expert_assign_correct += 1
                expert_assign_total += 1
                image_idx_counter += 1

                correct_expert_assignment = []
                for c_index in range(args.num_clients):
                    if ori_target[image_idx] in client_class_indices[c_index, :].tolist():
                        correct_expert_assignment.append(c_index)
                
                for cea in correct_expert_assignment:
                    perfect_digit_queues[cea].append(data[image_idx, :, :, :])
                    perfect_label_queues[cea].append(target[image_idx].data)

        logger.info("** Expert Selection Accuracy: {}/{}, {:.2f}%".format(expert_assign_correct, expert_assign_total, 
                                                        expert_assign_correct/expert_assign_total*100.0))

        expert_indices = []
        correct_labels = []
        for expert_idx, (digits, labels) in enumerate(zip(digit_queues, label_queues)):
            reconstructed_data = torch.stack(digits, dim=0).to(device)
            reconstructed_targets = torch.stack(labels, dim=0).to(device)
            net_local_list[expert_idx].eval()
            output = net_local_list[expert_idx](reconstructed_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(reconstructed_targets.view_as(pred)).sum().item()
            total += reconstructed_targets.size(0)

            # expert_indices.extend([expert_idx for _ in labels])
            # if args.dataset == "cifar10":
            #     correct_labels.extend([l.item() for l in labels])
            # elif args.dataset == "cifar100":
            #     tempt_cls = []
            #     for l in labels:
            #         for item_id, (k, v) in enumerate(client_sub_classes.items()):
            #             if l in v:
            #                 tempt_cls.append(k)
            #     correct_labels.extend(tempt_cls)
            # else:
            #     raise NotImplementedError("Unsupported Dataset ...")
    
        perfect_correct = 0
        perfect_total = 0
        for expert_idx, (digits, labels) in enumerate(zip(perfect_digit_queues, perfect_label_queues)):
            reconstructed_data = torch.stack(digits, dim=0).to(device)
            reconstructed_targets = torch.stack(labels, dim=0).to(device)
            net_local_list[expert_idx].eval()
            output = net_local_list[expert_idx](reconstructed_data)
            pred = output.argmax(dim=1, keepdim=True)
            perfect_correct += pred.eq(reconstructed_targets.view_as(pred)).sum().item()
            perfect_total += reconstructed_targets.size(0)

            # if args.dataset == "cifar10":
            #     correct_labels.extend([l.item() for l in labels])
            # elif args.dataset == "cifar100":
            #     tempt_cls = []
            #     for l in labels:
            #         for item_id, (k, v) in enumerate(client_sub_classes.items()):
            #             if l in v:
            #                 tempt_cls.append(k)
            #     correct_labels.extend(tempt_cls)
            # else:
            #     raise NotImplementedError("Unsupported Dataset ...")

        #conf_mat = confusion_matrix(y_true=expert_indices, y_pred=correct_labels)
        #logger.info("@@ Final {}/{}, Accuracy: {:.2f}%, Perfect Acc.: {:.2f}%,, conf. mat.: \n{}\n".format(
        #        correct, total, correct/total*100.0, perfect_correct/perfect_total*100.0, conf_mat))
        logger.info("@@ Final {}/{}, Accuracy: {:.2f}%, Perfect Acc.: {:.2f}%".format(
               correct, total, correct/total*100.0, perfect_correct/perfect_total*100.0))


def global_inference_learning_based(net_local_list, test_loader, args, client_class_indices, classifier, sampled_expert_indices):
    test_loss = 0
    correct = 0
    total = 0
    device = get_device()

    assert args.dataset == "cifar100"

    # sampled_expert_indices = np.random.choice(
    #                                 np.arange(args.num_clients), args.num_used_experts, 
    #                                 replace=False)

    if args.pred_mode == "label-pred":
        with torch.no_grad():
            for batch_dix, (data, target) in enumerate(test_loader):
                target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
                data = data.to(device)

                new_test = np.zeros((args.var_bs, int(args.sampled_experts*20)))
                client_shift_idx = 0
                for net_local_index, net_local in enumerate(net_local_list):
                    if net_local_index in sampled_expert_indices:
                        output = net_local(data)
                        smax = to_np(F.softmax(output, dim=1))
                        new_test[:, 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                        client_shift_idx+=1

                new_test = torch.from_numpy(new_test).float().to(device)
                output = classifier(new_test)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            logger.info("@@ Final {}/{}, Accuracy: {:.2f}%".format(
                            correct, total, correct/total*100.0))
    elif args.pred_mode == "expert-pred":
        with torch.no_grad():
            expert_assignment_list = []
            for batch_dix, (data, target) in enumerate(test_loader):
                target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
                data = data.to(device)

                new_test = np.zeros((args.var_bs, int(args.sampled_experts*20)))
                client_shift_idx = 0
                for net_local_index, net_local in enumerate(net_local_list):
                    if net_local_index in sampled_expert_indices:
                        output = net_local(data)
                        smax = to_np(F.softmax(output, dim=1))
                        new_test[:, 20*client_shift_idx:20*(client_shift_idx+1)] = smax
                        client_shift_idx+=1

                new_test = torch.from_numpy(new_test).float().to(device)
                output = classifier(new_test)
                pred = output.argmax(dim=1, keepdim=True)
                expert_assignment_list.extend(to_np(pred).flatten().tolist())

            # queuing inference into batches, and wire to appropriate expert network
            digit_queues = [[] for _ in range(args.num_clients)]
            label_queues = [[] for _ in range(args.num_clients)]

            perfect_digit_queues = [[] for _ in range(args.num_clients)]
            perfect_label_queues = [[] for _ in range(args.num_clients)] 

            image_idx_counter = 0
            expert_assign_correct = 0
            expert_assign_total = 0
            for data, target in test_loader:
                ori_target = target
                target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
                for image_idx in range(args.var_bs):
                    image_exprt_assignment = expert_assignment_list[image_idx_counter]
                    digit_queues[image_exprt_assignment].append(data[image_idx, :, :, :])
                    label_queues[image_exprt_assignment].append(target[image_idx].data)

                    if ori_target[image_idx] in client_class_indices[image_exprt_assignment, :].tolist():
                        expert_assign_correct += 1
                    expert_assign_total += 1
                    image_idx_counter += 1

                    correct_expert_assignment = []
                    for c_index in range(args.num_clients):
                        if ori_target[image_idx] in client_class_indices[c_index, :].tolist():
                            correct_expert_assignment.append(c_index)
                    
                    for cea in correct_expert_assignment:
                        perfect_digit_queues[cea].append(data[image_idx, :, :, :])
                        perfect_label_queues[cea].append(target[image_idx].data)
            logger.info("** Expert Selection Accuracy: {}/{}, {:.2f}%".format(expert_assign_correct, expert_assign_total, 
                                                            expert_assign_correct/expert_assign_total*100.0))

            expert_indices = []
            correct_labels = []
            for expert_idx, (digits, labels) in enumerate(zip(digit_queues, label_queues)):
                reconstructed_data = torch.stack(digits, dim=0).to(device)
                reconstructed_targets = torch.stack(labels, dim=0).to(device)
                net_local_list[expert_idx].eval()
                output = net_local_list[expert_idx](reconstructed_data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(reconstructed_targets.view_as(pred)).sum().item()
                total += reconstructed_targets.size(0)

            perfect_correct = 0
            perfect_total = 0
            for expert_idx, (digits, labels) in enumerate(zip(perfect_digit_queues, perfect_label_queues)):
                reconstructed_data = torch.stack(digits, dim=0).to(device)
                reconstructed_targets = torch.stack(labels, dim=0).to(device)
                net_local_list[expert_idx].eval()
                output = net_local_list[expert_idx](reconstructed_data)
                pred = output.argmax(dim=1, keepdim=True)
                perfect_correct += pred.eq(reconstructed_targets.view_as(pred)).sum().item()
                perfect_total += reconstructed_targets.size(0)

            logger.info("@@ Final {}/{}, Accuracy: {:.2f}%, Perfect Acc.: {:.2f}%".format(
                    correct, total, correct/total*100.0, perfect_correct/perfect_total*100.0))
    else:
        raise NotImplementedError("Unsupported prediction mode ...")


def global_inference_vanilla_ensemble(net_local_list, test_loader, args):
    test_loss = 0
    correct = 0
    total = 0
    device = get_device()

    assert args.dataset == "cifar100"
    client_sub_classes = {}
    for client_index in range(args.num_clients):
        sub_class_indices = [np.array_split(
                                np.where(CIFAR100_COARSE_LABELS == sc_index)[0], 
                                args.num_clients)[client_index].tolist() for sc_index in range(20)]
        client_sub_classes[client_index] = list(itertools.chain(*sub_class_indices))

    assert len(net_local_list) == args.num_clients
    with torch.no_grad():
        for batch_dix, (data, target) in enumerate(test_loader):
            target = torch.from_numpy(CIFAR100_COARSE_LABELS[target]).to(device)
            data = data.to(device)

            output_prob = torch.zeros(args.var_bs, 20).to(device)

            # the ensemble step
            for net_local_index, net_local in enumerate(net_local_list):
                output = net_local(data)

                smax = F.softmax(output, dim=1)
                output_prob += smax

            output_prob /= args.num_clients
            pred = output_prob.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        logger.info("@@ Final {}/{}, Accuracy: {:.2f}%".format(
                correct, total, correct/total*100.0))