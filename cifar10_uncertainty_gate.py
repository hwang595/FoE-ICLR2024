# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.
import argparse
import copy
import itertools
import logging
import copy

import torch

from timm.data import Mixup # to try Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from models import VGG, MCDropoutVGG, ResNet18, MCDropoutResNet18
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

from utils.sampling import *
from utils.network_utils import param_counter
from utils.reliability_graph import draw_reliability_graph
from utils.temperature_scaling import ModelWithTemperature
from utils.local_uncertainty import local_uncertainty
from utils.utils import get_dataset, get_device, save_network_logits, get_loaders, seed, \
CIFAR100_COARSE_LABELS, CIFAR100_SUBCLASSES

from src.validation_funcs import test_local, cross_expert_accs, cross_expert_confs, oracle_classifier, \
tree_oracle_classifier, cross_expert_classifier, global_inference_score_based, global_inference_mutual_conf, \
global_inference_learning_based, global_inference_vanilla_ensemble, shortest_path_approach, frugal_ml_logger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Non-iid CIFAR FL-MoE Training with public dataset to train gating.')
# FL
parser.add_argument('--num_clients', type=int, default=10, help="number of clients") # this is for testing how over method better leverages users' data
parser.add_argument('--num_total_clients', type=int, default=10, help="number of total clients") # this is for testing how over method better leverages users' data
parser.add_argument('--fl_rounds', type=int, default=20, help="FL training rounds")
parser.add_argument('--local_bs', type=int, default=128, help="batch size for local training")
parser.add_argument('--local_ep', default=200, type=int, help="local training epochs")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate for both local and global update")
parser.add_argument('--lr_decay_factor', default=0.995, type=float, help="learning rate decay factor.")
parser.add_argument('--momentum', default=0.9, type=float, help="momentum for both local and global update")
parser.add_argument('--test_bs', default=100, type=int, help="test batch size")
parser.add_argument('--T', default=0.1, type=float, help='temperature: energy|Odin')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

# dataset
parser.add_argument('--dataset', type=str, default="cifar10", help="the dataset to use for the experiment.")
parser.add_argument('--par-strategy', type=str, default="corse-label", 
                                      help="how to partition the dataset: |corse-label|sample-based|.")

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# eval mode
parser.add_argument('--eval-mode', type=str, default="score-based",
                    help='mode for evaluation: |score-based|mutual-info| .')
#parser.add_argument('--num-used-experts', type=int, default=20,
#                    help='with in the range of 1 to number of total clients .')
parser.add_argument('--pred-mode', type=str, default="label-pred",
                    help='mode for prediction in learning based: |label-pred|expert-pred| .')
parser.add_argument('--e-mat-score', type=str, default="maxsoftmax",
                    help='mode for calculating E matrix: |maxsoftmax|entropy| .')
parser.add_argument('--dist-metric', type=str, default="cosine",
                    help='mode for evaluation: |cosine|l2| .')
parser.add_argument('--use-histgram', action='store_true',
                    help='use histgram to store E matrix .')

# model fuser args
parser.add_argument('--sampled-experts', type=int, default=20,
                    help='how many experts feat sampled for the learning based fusing .')
parser.add_argument('--sp-model', type=str, default="nn",
                    help='sp model to use for shortest path classifier, options |"nn"|"knn"|')
parser.add_argument('--beam-search', action='store_true',
                    help='the option of beam search for nn to avoid training too many NNs.')
parser.add_argument('--num-apis-per-layer-bs', type=int, default=5,
                    help='the number of APIs to select for each layer of beam search.')
# shortest path approach args
parser.add_argument('--k4knn', type=int, default=10,
                    help='k in kNN used for estimating distance.')
parser.add_argument('--delta-dist-threshold', type=float, default=2.0,
                    help='k in kNN used for estimating distance.')

# utils
parser.add_argument('--save_logits', action='store_true',
                    help='save logits of trained local experts .')
parser.add_argument('--resume', action='store_true',
                    help='resum from trained expert networks .')
parser.add_argument('--seed', type=int, default=42,
                    help='resum from trained expert networks .')

# MoE
parser.add_argument('--k', default=4, type=int, help="select top k")
# parser.add_argument('--num_experts', default=10, type=int, help="number of experts")
parser.add_argument('--hidden_size', default=256, type=int, help="hidden size")
parser.add_argument('--var_iter_budget', type=int, default=400,
                    help='# iterations for measuring prediction variance during the inference time.')
parser.add_argument('--var_bs', type=int, default=1,
                    help='batch size for measuring prediction variance.')
parser.add_argument('--opt_name', default='sgd', type=str,
                    help='which type of calibration to use.')
parser.add_argument('--caliration_type', default='temp-scaling', type=str,
                    help='which type of calibration to use.')
args = parser.parse_args()


def feat_map_avg(feat_map_list):
    feat_map_avg = []
    for p_index, p in enumerate(feat_map_list[0].parameters()):
        avg_weight = torch.zeros_like(p)
        for local_net_id in range(len(feat_map_list)):
            local_feat_map_params = list(feat_map_list[local_net_id].parameters())
            avg_weight += local_feat_map_params[p_index].data
        feat_map_avg.append(avg_weight/len(feat_map_list))
    return feat_map_avg


class LocalUpdatePub(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args

        #self.trainloader = self.get_loaders(dataset=dataset, idxs=idxs, valid=False)
        self.trainloader, self.valoader, _ = get_loaders(
                                                dataset=dataset, 
                                                idxs=idxs, 
                                                valid=True,
                                                args=args)

        # setting mixup here
        self.mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        if mixup_active:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def get_optimizer(self, net, opt_name="sgd"):
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), 
                                    lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-4)
            scheduler = MultiStepLR(optimizer, 
                                    milestones=[int(0.5*self.args.local_ep), int(0.75*self.args.local_ep)], 
                                    gamma=0.1)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(net.parameters(), 
                                      lr=self.args.lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_ep)
        else:
            raise NotImplementedError("Unsupported Optimizer Type...")
        return optimizer, scheduler        

    def train(self, net, client_idx, fl_round=0):
        # only update the client's local expert and gating function
        net.train()

        optimizer, scheduler = self.get_optimizer(net=net, opt_name=self.args.opt_name)
        scaler = GradScaler()

        epoch_loss = []
        for epoch in range(args.local_ep):
            batch_loss = []
            for i, data in enumerate(self.trainloader):
                inputs, labels = data

                if self.args.dataset == "cifar100":
                    # we conduct coarse label classification here
                    labels = torch.from_numpy(CIFAR100_COARSE_LABELS[labels])
                inputs, labels = inputs.to(device), labels.to(device)

                if self.mixup_fn is not None:
                    inputs, labels = self.mixup_fn(inputs, labels)

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    outputs = net(inputs)
                    loss = self.criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if epoch % 10 == 0:
                logger.info('Client: {}, FL Round {:3d}, Local train loss {:.3f}, Num batches: {}, LR: {}'.format(
                        client_idx, fl_round, epoch_loss[-1], len(self.trainloader), scheduler.get_last_lr()))
        return sum(epoch_loss) / len(epoch_loss)


# trainset, testset, valset, args.nb_classes = get_dataset(ds="CIFAR10", validation_split=True)
# valoader = torch.utils.data.DataLoader(valset,
#                                         batch_size=args.test_bs,
#                                         shuffle=False,
#                                         num_workers=4,
#                                         pin_memory=True) 

seed(seed=args.seed)
trainset, testset, args.nb_classes = get_dataset(ds=args.dataset, validation_split=False)
trainloader = torch.utils.data.DataLoader(trainset,
                                        batch_size=args.local_bs,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True)
testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=args.test_bs,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True)
testloader_global = torch.utils.data.DataLoader(testset,
                                        batch_size=args.var_bs,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True)
# non-iid split for local udpate
if args.dataset == "cifar10":
    train_id_list, client_class_indices = cifar_synthetic(trainset, args.num_clients)
elif args.dataset == "cifar100":
    if args.par_strategy == "corse-label":
        train_id_list, client_class_indices = cifar_coarse_label(trainset, args.num_clients)
    elif args.par_strategy == "sample-based":
        train_id_list, client_class_indices = cifar_sample_label(trainset, args.num_clients, classes_per_parti=30)
#train_id_list, train_cls_weight = cifar_noniid(trainset, args.num_total_clients, alpha=0.5)

device = get_device()
net_local_list = [ResNet18(num_classes=args.nb_classes).to(device)
                    for client_index in range(args.num_clients)
                ]
logger.info("@ Local Model: {}, num_params: {}".format(
                                            net_local_list[0], param_counter(net_local_list[0])))

for fl_round in range(args.fl_rounds):    
    # local training procedure
    for local_id in range(args.num_clients):
        if not args.resume:
            local = LocalUpdatePub(args, trainset, train_id_list[local_id])
            loss = local.train(net=net_local_list[local_id], client_idx=local_id, fl_round=fl_round)

            # reliability graph plotting
            preds_sm, labels_oneh = test_local(net_local_list[local_id], testloader, 
                                                client_idx=local_id, 
                                                client_class_indices=client_class_indices[local_id, :], 
                                                args=args)

            draw_reliability_graph(outputs=preds_sm, 
                                    labels=labels_oneh, 
                                    net_id=local_id, 
                                    prefix="before_clib")

            if args.caliration_type == "temp-scaling":
                net_local_list[local_id] = ModelWithTemperature(net_local_list[local_id])
                #net_local_list[local_id].set_temperature(valoader)
                net_local_list[local_id].set_temperature(local.valoader)
                preds_sm, labels_oneh = test_local(net_local_list[local_id], testloader, 
                                                    client_idx=local_id, args=args
                                                    )
                draw_reliability_graph(outputs=preds_sm, 
                                        labels=labels_oneh, 
                                        net_id=local_id, 
                                        prefix="after_temp_scale")

            if args.save_logits:
                save_network_logits(net_local_list[local_id], local.trainloader, 
                                file_name="trainset_logits_client{}".format(local_id), 
                                args=args)
                save_network_logits(net_local_list[local_id], testloader, 
                                file_name="testset_logits_client{}".format(local_id), 
                                args=args)
            torch.save(net_local_list[local_id].state_dict(), "expert-{}_ckpt".format(local_id))
        else:
            net_local_list[local_id].load_state_dict(torch.load("expert-{}_ckpt".format(local_id)))
            # reliability graph plotting
            preds_sm, labels_oneh = test_local(net_local_list[local_id], testloader, 
                                            client_idx=local_id, 
                                            client_class_indices=client_class_indices[local_id, :], 
                                            args=args)

    #if args.dataset == "cifar100":
    #    cross_expert_accs(net_local_list, testloader, client_class_indices, args)

    if args.eval_mode == "mutual-info":
        train_loaders = []
        for local_id in range(args.num_clients):
            trainloader, valoader, _ = get_loaders(dataset=trainset, idxs=train_id_list[local_id], valid=True, args=args)
            train_loaders.append(trainloader)
        E = cross_expert_confs(net_local_list, train_loaders, args)
        global_inference_mutual_conf(net_local_list, testloader_global, args=args, 
                                      client_class_indices=client_class_indices, cross_client_info=E)

        with open("E_matrix_{}".format(args.dist_metric)+".npy","wb") as file_to_save:
            np.save(file_to_save, E)
    elif args.eval_mode == "score-based":
        # global inference
        global_inference_score_based(net_local_list, testloader_global, args=args, 
                                      client_class_indices=client_class_indices, score="softmax")
    elif args.eval_mode == "learning-based":
        val_data_list = []
        for local_id in range(args.num_clients):
            local_data_indices = train_id_list[local_id]
            _, local_valoader, _ = get_loaders(dataset=trainset, 
                                            idxs=local_data_indices, 
                                            valid=True,
                                            args=args)
            val_data_list.append(local_valoader.dataset)
        valloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_data_list), batch_size=args.local_bs, 
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True)
        classifier, sampled_expert_list = cross_expert_classifier(net_local_list, 
                                        trainloader=valloader, args=args)
        global_inference_learning_based(net_local_list, testloader_global, args=args, 
                                      client_class_indices=client_class_indices, classifier=classifier,
                                      sampled_expert_indices=sampled_expert_list)
    elif args.eval_mode == "learning-based-oracle":
        val_data_list = []
        #verify_data_list = []
        for local_id in range(args.num_clients):
            local_data_indices = train_id_list[local_id]
            _, local_valoader, _ = get_loaders(dataset=trainset, 
                                            idxs=local_data_indices, 
                                            valid=True,
                                            args=args)
            val_data_list.append(local_valoader.dataset)
        valloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_data_list), batch_size=args.local_bs, 
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True)
        classifier, sampled_expert_list = oracle_classifier(net_local_list, 
                                        trainloader=valloader, testloader=testloader_global, args=args)
    elif args.eval_mode == "tree-oracle":
        val_indices_list = []
        val_dataset_list = []
        for local_id in range(args.num_clients):
            local_data_indices = train_id_list[local_id]
            _, local_valoader, valid_idxs = get_loaders(dataset=trainset, 
                                            idxs=local_data_indices, 
                                            valid=True,
                                            args=args)
            idx = np.array(local_valoader.dataset.idxs)
            #val_data_list.append(local_valoader.dataset.dataset.data[idx])
            #val_target_list.append(np.array(local_valoader.dataset.targets)[idx])
            val_indices_list.append(valid_idxs)
            val_dataset_list.append(local_valoader.dataset)
        #val_data, val_targets = np.concatenate(val_data_list, axis=0), np.concatenate(val_target_list, axis=0) 
        val_indices = np.concatenate(val_indices_list)
        classifier, sampled_expert_list = tree_oracle_classifier(net_local_list, 
                                        testloader=testloader_global,
                                        val_indices=val_indices,
                                        args=args)
    elif args.eval_mode == "shortest-path":
        val_indices_list = []
        val_dataset_list = []
        for local_id in range(args.num_clients):
            local_data_indices = train_id_list[local_id]
            _, local_valoader, valid_idxs = get_loaders(dataset=trainset, 
                                            idxs=local_data_indices, 
                                            valid=True,
                                            args=args)
            idx = np.array(local_valoader.dataset.idxs)
            #val_data_list.append(local_valoader.dataset.dataset.data[idx])
            #val_target_list.append(np.array(local_valoader.dataset.targets)[idx])
            val_indices_list.append(valid_idxs)
            val_dataset_list.append(local_valoader.dataset)
        #val_data, val_targets = np.concatenate(val_data_list, axis=0), np.concatenate(val_target_list, axis=0) 
        val_indices = np.concatenate(val_indices_list)
        classifier, sampled_expert_list = shortest_path_approach(
                                                net_local_list, 
                                                val_indices=val_indices, 
                                                testloader=testloader_global, 
                                                args=args
                                            )        
    elif args.eval_mode == "frugal-ml":
        val_data_list = []
        for local_id in range(args.num_clients):
            local_data_indices = train_id_list[local_id]
            _, local_valoader, _ = get_loaders(dataset=trainset, 
                                            idxs=local_data_indices, 
                                            valid=True,
                                            args=args)
            val_data_list.append(local_valoader.dataset)
        # we will also need to add testset here
        val_data_list.append(testloader_global.dataset)
        valloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_data_list), batch_size=args.local_bs, 
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True)
        frugal_ml_logger(net_local_list, trainloader=valloader, args=args) 
    elif args.eval_mode == "ensemble":
        global_inference_vanilla_ensemble(net_local_list, testloader_global, args)
    else:
        raise NotImplementedError("Unsupported eval model ...")
    
    exit()
    args.lr = args.lr_decay_factor * args.lr

logger.info('Finished Federated Training')