import json
import copy

import torch
import argparse
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPTQConfig
from peft import PeftModel, PeftConfig

#from utils import MMLU_AVAIL_CATEGORIES, seed, get_logger, to_np
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='MMLU Gating.')
    parser.add_argument('--seed', type=int, default=42,
                        help='resum from trained expert networks .')
    parser.add_argument('--input-bs', type=int, default=4,
                        help='resum from trained expert networks .')
    parser.add_argument('--start-model-id', type=int, default=0,
                        help='resum from trained expert networks .')
    parser.add_argument('--cons-train', action='store_true',
                    help='build data for training .')
    parser.add_argument('--cons-eval', action='store_true',
                    help='build data for evaluation .')
    parser.add_argument('--cons-test', action='store_true',
                    help='build data for testing .')
    args = parser.parse_args()
    return args

def get_model_names():
    logger = get_logger()
    mmlu_df = pd.read_csv('model_evaluation_results.csv')
    cols_mmlu = [i for i, c in enumerate(mmlu_df.columns) if c.startswith('MMLU') and c != 'MMLU_average']
    mmlu_names = [c for i, c in enumerate(mmlu_df.columns) if c.startswith('MMLU') and c != 'MMLU_average']
    logger.info("* MMLU categories: {}".format(mmlu_names))
    all_model_names = mmlu_df['Model Name'].values.tolist()

    acc_thr = 0.5
    filter_ = mmlu_df['MMLU_average'] <= acc_thr
    acc_mmlu = mmlu_df.values[:,cols_mmlu].astype(float)[filter_]
    logger.info('Best overall : {}'.format(acc_mmlu.mean(axis=1).max()))
    logger.info('Best oracle : {}'.format(acc_mmlu.max(axis=0).mean()))

    best_models = []
    for d, i in enumerate(acc_mmlu.argmax(axis=0)):
        #print("Best Model Name: {}, Model Params: {}B, Category Name: {}".format(
        #        mmlu_df['Model Name'].values[filter_][i], mmlu_df['Parameters'].values[filter_][i], mmlu_names[d]
        #    ))
        _model_size, _model_name = mmlu_df['Parameters'].values[filter_][i], mmlu_df['Model Name'].values[filter_][i]
        # hwang: it seems to be hard to get GPTQ thing work in cuda12.0, will look at this later
        if not np.isnan(_model_size) and _model_size < 10 and _model_name not in ("WizardLM-13B-V1.1-GPTQ", "trurl-2-7b", "orca_mini_v2_13b"):
            best_models.append(
                                "{}/{}".format(mmlu_df['organization'].values[filter_][i], 
                                mmlu_df['Model Name'].values[filter_][i])
                            )
    model_names = np.sort(list(set(best_models))).tolist()
    logger.info('Unique experts : {}'.format(len(set(best_models))))

    category_true_labels = {}
    for mmlu_cate_id in cols_mmlu:
        category_scores = []
        # we are going to construct label here
        # for each category in MMLU, we assign the label with the highest score
        for mn in model_names:
            model_name = mn.split("/")[-1]
            category_scores.append(mmlu_df.iloc[all_model_names.index(model_name), mmlu_cate_id])
        max_id = np.argmax(category_scores)
        category_true_labels[mmlu_df.columns[mmlu_cate_id].lstrip("MMLU_")] = max_id
        logger.info("@ Cate name: {}, Best Model: {}, True Label: {}".format(
                                                mmlu_df.columns[mmlu_cate_id], model_names[max_id], max_id
                                            ))
    logger.info("!! Category_true_labels: {}, Unique true labels: {}, len unqie: {}".format(
                                            category_true_labels, set(category_true_labels), 
                                            len(set(category_true_labels))
                                        ))
    logger.info("!! category_true_labels keys: {}".format(category_true_labels.keys()))
    return model_names, category_true_labels


def get_ori_datasets():
    logger = get_logger()
    train_datasets, eval_datasets, test_datasets = {}, {}, {}
    num_dps_train, num_dps_eval, num_dps_test = 0, 0, 0
    for mmlu_category in MMLU_AVAIL_CATEGORIES:
        train_datasets[mmlu_category] = load_dataset("lukaemon/mmlu", mmlu_category, split="train")
        eval_datasets[mmlu_category] = load_dataset("lukaemon/mmlu", mmlu_category, split="validation")
        test_datasets[mmlu_category] = load_dataset("lukaemon/mmlu", mmlu_category, split="test")
        num_dps_train += len(train_datasets[mmlu_category]["input"])
        num_dps_eval += len(eval_datasets[mmlu_category]["input"])
        num_dps_test += len(test_datasets[mmlu_category]["input"])
    logger.info("!!! Num DPS train: {}, Num DPS eval: {}, Num DPS test: {}".format(
                num_dps_train, num_dps_eval, num_dps_test
        ))
    return train_datasets, eval_datasets, test_datasets, num_dps_train, num_dps_eval, num_dps_test


def get_calibrated_labels(datasets, category_true_labels, num_dps, model_names, args):

    reconstructed_label = np.zeros(num_dps)
    logger = get_logger()

    recons_data_bias = 0 # global_batch_counter
    for model_id, mn in enumerate(model_names):
        gb_start_idx = 0
        with torch.no_grad():
            for data_id, (category_name, data_class) in enumerate(datasets.items()):
                data_item = data_class["input"]
                
                logger.info("category name: {}, len data class: {}".format(category_name, len(data_item)))

                category_rec_labels = np.zeros(len(data_item))
                for data_idx, data_input in enumerate(data_item):
                    logger.info("** recons label: {}".format(category_true_labels[category_name]))
                    category_rec_labels[data_idx] = category_true_labels[category_name]

                logger.info("-- beg: {}, end: {} --".format(gb_start_idx, gb_start_idx+len(data_item)))
                if model_id == 0:
                    reconstructed_label[gb_start_idx:gb_start_idx+len(data_item)] = category_rec_labels
                gb_start_idx += len(data_item)
    return reconstructed_label


if __name__ == "__main__":
    args, logger = get_args(), get_logger()
    # set seed
    logger.info("seeding ...")
    seed(args.seed)

    embed_dims = [4096, 4096, 2560, 4096, 4096, 64, 4096, 4096, 
                    4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    test_bs = 10

    model_names, category_true_labels = get_model_names()
    logger.info("@@@@ model names: {}".format(model_names))

    train_datasets, eval_datasets, test_datasets, num_dps_train, num_dps_eval, num_dps_test = get_ori_datasets()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_list = []
    tokenizer_list = []

    with open('overall_apis_category_acc_collector.json', 'r') as infile:
        overall_category_acc_collector = json.load(infile)

    calibrated_category_true_labels = {}
    list_api_labels, oracle_acc, acc_state = [], {}, {}
    model_names.remove("THUDM/chatglm2-6b")
    for category_id, category_name in enumerate(category_true_labels.keys()):
        logger.info("category id: {}, category_name: {}".format(category_id, category_name))
        category_scores = []
        for model_id, mn in enumerate(model_names):
            category_scores.append(overall_category_acc_collector[mn][category_name])
        #logger.info("category scores: {}, max id: {}, max mn: {}".format(
        #                            category_scores, np.argmax(category_scores), model_names[np.argmax(category_scores)]))
        calibrated_category_true_labels[category_name] = int(np.argmax(category_scores))
        oracle_acc[category_name] = max(category_scores)
        acc_state[category_name] = {"mean":np.mean(category_scores), "std":np.std(category_scores)}
        list_api_labels.append(int(np.argmax(category_scores)))

    logger.info("** Oracle Cate Accs: {}, \n \n Cate Acc Stats: {}".format(
                                                oracle_acc, acc_state
                                            ))
    # reconstruct oracle accuracy here
    oracle_total, oracle_correct = 0, 0
    for data_id, (category_name, data_class) in enumerate(test_datasets.items()):
        oracle_correct += int((len(data_class)*oracle_acc[category_name]/100.0))
        oracle_total += len(data_class)
    logger.info("** Oracle classifier: {}/{}, Acc: {}".format(
                                oracle_correct, oracle_total, (oracle_correct/oracle_total)*100.0
                            ))

    logger.info("** Calibrated category true labels: {}".format(calibrated_category_true_labels))
    logger.info("** List true labels: {}, unique models: {}".format(list_api_labels, len(set(list_api_labels))))
    with open("calibrated_category_true_labels.json", "w") as j_out:
        json.dump(calibrated_category_true_labels, j_out)
    
    reconstructed_label_train = get_calibrated_labels(train_datasets, calibrated_category_true_labels, num_dps_train, model_names, args)
    reconstructed_label_eval = get_calibrated_labels(eval_datasets, calibrated_category_true_labels, num_dps_eval, model_names, args)
    reconstructed_label_test = get_calibrated_labels(test_datasets, calibrated_category_true_labels, num_dps_test, model_names, args)

    #reconstructed_dataset_train= np.load('reconstructed_dataset_ckpt_mn7_train.npy')[:, 0:sum(embed_dims[0:1])] # api 1 solely
    #reconstructed_dataset_train= np.load('reconstructed_dataset_ckpt_mn7_train.npy')[:, embed_dims[0]:sum(embed_dims[0:2])] # api 2 solely
    #reconstructed_dataset_train= np.load('reconstructed_dataset_ckpt_mn7_train.npy')[:, sum(embed_dims[0:2]):sum(embed_dims[0:3])] # api 3 solely
    reconstructed_dataset_train= np.load('reconstructed_dataset_ckpt_mn7_train.npy')[:, 0:sum(embed_dims[0:3])] # api 1+2+3
    reconstructed_label_train = copy.deepcopy(reconstructed_label_train)

    #reconstructed_dataset_val= np.load('reconstructed_dataset_eval.npy')[:, 0:sum(embed_dims[0:1])] # api 1 solely
    #reconstructed_dataset_val= np.load('reconstructed_dataset_eval.npy')[:, embed_dims[0]:sum(embed_dims[0:2])] # api 2 solely
    #reconstructed_dataset_val= np.load('reconstructed_dataset_eval.npy')[:, sum(embed_dims[0:2]):sum(embed_dims[0:3])] # api 3 solely
    reconstructed_dataset_val= np.load('reconstructed_dataset_eval.npy')[:, 0:sum(embed_dims[0:3])]
    reconstructed_label_val = copy.deepcopy(reconstructed_label_eval)

    #reconstructed_dataset_test= np.load('reconstructed_dataset_ckpt_mn2_test.npy')[:, 0:sum(embed_dims[0:1])] # api 1 solely
    #reconstructed_dataset_test= np.load('reconstructed_dataset_ckpt_mn2_test.npy')[:, embed_dims[0]:sum(embed_dims[0:2])] # api 2 solely
    #reconstructed_dataset_test= np.load('reconstructed_dataset_ckpt_mn2_test.npy')[:, sum(embed_dims[0:2]):sum(embed_dims[0:3])] # api 3 solely
    reconstructed_dataset_test= np.load('reconstructed_dataset_ckpt_mn2_test.npy')[:, 0:sum(embed_dims[0:3])]
    reconstructed_label_test = copy.deepcopy(reconstructed_label_test)

    # print("reconstructed val dataset size: {}, reconstructed val label size: {}".format(
    #         reconstructed_dataset_val.shape, reconstructed_label_val.shape
    #     ))
    # print("reconstructed val dataset: {}, reconstructed val label: {}".format(
    #         reconstructed_dataset_val, reconstructed_label_val
    #     ))
    # print("reconstructed train dataset size: {}, reconstructed train label size: {}".format(
    #         reconstructed_dataset_train.shape, reconstructed_label_train.shape
    #     ))
    # print("reconstructed train dataset: {}, reconstructed train label: {}".format(
    #         reconstructed_dataset_train, reconstructed_label_train
    #     ))
    reconstructed_dataset_val = np.concatenate((reconstructed_dataset_train, reconstructed_dataset_val), axis=0)
    reconstructed_label_val = np.concatenate((reconstructed_label_train, reconstructed_label_val), axis=0)
    logger.info("New val dataset size: {}, New val label size: {}".format(
            reconstructed_dataset_val.shape, reconstructed_label_val.shape
        ))
    logger.info("reconstructed test dataset size: {}, reconstructed test label size: {}".format(
            reconstructed_dataset_test.shape, reconstructed_label_test.shape
        ))
    logger.info("reconstructed_dataset: {}, reconstructed_label: {}".format(
            reconstructed_dataset_test, reconstructed_label_test
        ))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    classifier = SimpleClassifierForSentimentAnalysis(
                   num_clients=15, embedding_dims=embed_dims[0:3], hidden_dim=8192).to(device)

    logger.info("* Arch of the new classifier: {}".format(classifier))
    _new_classifier_eps = 50
    new_data_loader = torch.utils.data.DataLoader(SimpleDataset(data=reconstructed_dataset_val, 
                                                    targets=reconstructed_label_val),
                                                    batch_size=64, shuffle=True, num_workers=4)
    new_data_loader_test = torch.utils.data.DataLoader(SimpleDataset(data=reconstructed_dataset_test, 
                                                    targets=reconstructed_label_test),
                                                    batch_size=test_bs, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_classifier_eps)

    def eval(new_data_loader_test, classifier, ep, test_bs, model_assignment=None):
        test_loss, correct, total = 0, 0, 0
        classifier.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(new_data_loader_test):
                data, target = data.float().to(device), target.long().to(device)
                # the ensemble step
                outputs = classifier(data)
                _, predicted = outputs.max(1)
                if model_assignment is not None:
                    model_assignment[batch_idx*test_bs:(batch_idx+1)*test_bs] = predicted.flatten().data.cpu().numpy()
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            print("@@ EP: {}, Final {}/{}, Accuracy: {:.2f}%".format(
                    ep, correct, total, correct/total*100.0))
        return model_assignment

    for ep in range(_new_classifier_eps):
        train_loss, correct, total = 0, 0, 0
        classifier.train()
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
                print("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                        ep,
                        batch_idx, len(new_data_loader), train_loss/(batch_idx+1), 
                        100.*correct/total, correct, total))
        if ep % 10 == 0:
            eval(new_data_loader_test, classifier, ep=ep, test_bs=test_bs)
        scheduler.step()
    api_assignment = np.zeros(reconstructed_label_test.shape)
    api_assignment = eval(new_data_loader_test, classifier, ep=ep, test_bs=test_bs, model_assignment=api_assignment)

    with open('api_assignment_api123_feat_calibrated.npy', 'wb') as f:
        np.save(f, api_assignment)
    logger.info("Fin ..")