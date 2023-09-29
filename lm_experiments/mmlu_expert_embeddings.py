import json

import torch
import argparse
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPTQConfig
from peft import PeftModel, PeftConfig

from utils import MMLU_AVAIL_CATEGORIES, seed, get_logger, to_np

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

def get_recons_datasets(datasets, category_true_labels, num_dps, model_names, hidden_sizes, device, args, start_model_id=0, mode='eval'):
    reconstructed_dataset = np.zeros((num_dps, sum(hidden_sizes)))
    reconstructed_label = np.zeros(num_dps)
    logger = get_logger()

    recons_data_bias = 0 # global_batch_counter
    for model_id, mn in enumerate(model_names):
        gb_start_idx = 0
        logger.info("^^^ loading model: {}".format(mn))
        model = AutoModelForCausalLM.from_pretrained("{}".format(mn),
                                    torch_dtype=torch.float16, trust_remote_code=True
                                    )
        tokenizer = AutoTokenizer.from_pretrained("{}".format(mn), trust_remote_code=True)
        with torch.no_grad():
            for data_id, (category_name, data_class) in enumerate(datasets.items()):
                data_item = data_class["input"]
                if model_id >= start_model_id:
                    logger.info("category name: {}, len data class: {}".format(category_name, len(data_item)))
                    category_rec_ds = np.zeros((len(data_item), hidden_sizes[model_id]))
                    category_rec_labels = np.zeros(len(data_item))
                    for data_idx, data_input in enumerate(data_item):
                        model = model.to(device)
                        inputs = tokenizer(data_input, return_tensors="pt").to(device)
                        #logger.info("*** input size: {}".format(inputs["input_ids"].size()))
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                            content = model.generate(inputs["input_ids"], num_beams=1, do_sample=True,
                                        return_dict_in_generate=True, output_scores=True, 
                                        output_hidden_states=True, max_new_tokens=16)
                        mean_seq_embedding = to_np(torch.mean(content.hidden_states[-1][-1][0, :, :], dim=0))
                        category_rec_ds[data_idx, :] = mean_seq_embedding

                        # offload to CPU and try to save memory
                        model, inputs = model.to("cpu"), inputs.to("cpu")
                        torch.cuda.empty_cache()
                        
                        logger.info("** recons label: {}".format(category_true_labels[category_name]))
                        category_rec_labels[data_idx] = category_true_labels[category_name]

                    logger.info("-- beg: {}, end: {} --".format(gb_start_idx, gb_start_idx+len(data_item)))
                    reconstructed_dataset[gb_start_idx:gb_start_idx+len(data_item), recons_data_bias:recons_data_bias+hidden_sizes[model_id]] = category_rec_ds
                    if model_id == 0:
                        reconstructed_label[gb_start_idx:gb_start_idx+len(data_item)] = category_rec_labels
                gb_start_idx += len(data_item)
        if model_id >= start_model_id:
            # for each model we save a checkpoint
            with open('reconstructed_dataset_ckpt_mn{}_{}.npy'.format(model_id, mode), 'wb') as f:
                np.save(f, reconstructed_dataset)
            if model_id == 0:
                with open('reconstructed_label_ckpt_mn{}_{}.npy'.format(model_id, mode), 'wb') as f:
                    np.save(f, reconstructed_label)
        elif model_id == start_model_id - 1:
            logger.info(
                "Loading ckpt from : reconstructed_dataset_ckpt_mn{}_{}.npy & reconstructed_label_ckpt_mn{}.npy ".format(
                    start_model_id-1, 0))
            reconstructed_dataset = np.load('reconstructed_dataset_ckpt_mn{}_{}.npy'.format(start_model_id-1, mode))
            reconstructed_label = np.load('reconstructed_label_ckpt_mn{}.npy'.format(0, mode)) # here we assume we alway resume from start model id > 0
        recons_data_bias += hidden_sizes[model_id]
    return reconstructed_dataset, reconstructed_label

def eval_fuser_acc(test_datasets, category_true_labels, model_names):
    #block_model_name_list = ["THUDM/chatglm2-6b"] # block model name list (can't access their res on huggingface open leaderboard)
    # evaluate the acc of fuser
    total, correct, glboal_skipped_counter = 0, 0, 0
    category_wise_acc_collector = {}
    api_assignment = np.load("api_assignment_api1_feat_calibrated_v1.npy")
    for data_id, (category_name, data_class) in enumerate(test_datasets.items()):
        category_correct, category_skipped_counter = 0, 0
        data_item = data_class["input"]
        expert_model_name = model_names[category_true_labels[category_name]]
        logger.info("data_id: {}, category_name: {}, label: {}, model name: {}".format(
                                data_id, category_name, category_true_labels[category_name], model_names[category_true_labels[category_name]]
                            ))

        for data_idx, data_input in enumerate(data_item):
            logger.info("!! {}/{} in data category, global data ID: {}".format(data_idx, len(data_class), total))
            pred_api_id = int(api_assignment[total])
            org, real_mn = model_names[pred_api_id].split("/")

            result_data = load_dataset("open-llm-leaderboard/details_{}__{}".format(org, real_mn),
                                        "harness_hendrycksTest_{}_5".format(category_name), split="latest")
            for item_id, res_item in enumerate(result_data):
                if data_input in res_item["example"]:
                    correct += res_item["acc"]
                    category_correct += res_item["acc"]
                    break
            total += 1
        effective_data_num = len(data_class) - category_skipped_counter
        logger.info("*** Category: {}, Fuser Acc: {}/{}; {:.4f}%".format(
                                category_name, category_correct, effective_data_num, float(category_correct/effective_data_num)*100.0
                            ))
        category_wise_acc_collector[category_name] = float(category_correct/effective_data_num)*100.0
        glboal_skipped_counter += category_skipped_counter
    effective_global_data_num = total - glboal_skipped_counter
    logger.info("!!!! Fuser Acc: {}/{}; {:.4f}%, Category Breakdown: {}".format(
                correct, effective_global_data_num, float(correct/effective_global_data_num)*100.0, category_wise_acc_collector
            ))

def eval_ind_acc(test_datasets, category_true_labels, model_names, eval_model_id=0):
    #block_model_name_list = ["THUDM/chatglm2-6b"] # block model name list (can't access their res on huggingface open leaderboard)
    # evaluate the acc of fuser
    category_wise_acc_collector = {}
    org, real_mn = model_names[eval_model_id].split("/")
    expert_model_name = model_names[eval_model_id]
    
    total, correct, glboal_skipped_counter = 0, 0, 0
    for data_id, (category_name, data_class) in enumerate(test_datasets.items()):
        # if expert_model_name in block_model_name_list:
        #     logger.info("*** Skipping category: {}, as API : {} not on HF ...".format(
        #                                     category_name, expert_model_name
        #                                 ))
        #     total = 1e4 # to avoid divide by zero err
        #     continue
        result_data = load_dataset("open-llm-leaderboard/details_{}__{}".format(org, real_mn),
                                        "harness_hendrycksTest_{}_5".format(category_name), split="latest")
        category_correct, category_skipped_counter = 0, 0
        data_item = data_class["input"]
        
        logger.info("data_id: {}, category_name: {}, label: {}, model name: {}".format(
                                data_id, category_name, category_true_labels[category_name], model_names[category_true_labels[category_name]]
                            ))

        for data_idx, data_input in enumerate(data_item):
            logger.info("!! {}/{} in data category, total evaludate: {}".format(data_idx, len(data_class), total))
            
            # if model_names[eval_model_id] in block_model_name_list:
            #     logger.info("skipped data: {}, due to : {} not on HF".format(data_idx, model_names[pred_api_id]))
            #     category_skipped_counter += 1
            #     total += 1
            #     continue

            for item_id, res_item in enumerate(result_data):
                if data_input in res_item["example"]:
                    correct += res_item["acc"]
                    category_correct += res_item["acc"]
                    break
            total += 1
        effective_data_num = len(data_class) - category_skipped_counter
        logger.info("*** Category: {}, API: {} Acc: {}/{}; {:.4f}%".format(
                                category_name, eval_model_id, category_correct, effective_data_num, float(category_correct/effective_data_num)*100.0
                            ))
        category_wise_acc_collector[category_name] = float(category_correct/effective_data_num)*100.0
        glboal_skipped_counter += category_skipped_counter
    effective_global_data_num = total - glboal_skipped_counter
    logger.info("!!!! Model Name: {} API: {} Acc: {}/{}; {:.4f}%, Category Breakdown: {}".format(model_names[eval_model_id],
                    eval_model_id, correct, effective_global_data_num, float(correct/effective_global_data_num)*100.0, category_wise_acc_collector
                ))
    return category_wise_acc_collector

def get_calibrated_true_labels():
    with open('calibrated_category_true_labels.json', 'r') as infile:
        calibrated_category_true_labels = json.load(infile)

    return calibrated_category_true_labels

if __name__ == "__main__":
    args, logger = get_args(), get_logger()
    # set seed
    logger.info("seeding ...")
    seed(args.seed)

    model_names, category_true_labels = get_model_names()
    logger.info("@@@@ model names: {}".format(model_names))

    train_datasets, eval_datasets, test_datasets, num_dps_train, num_dps_eval, num_dps_test = get_ori_datasets()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_list = []
    tokenizer_list = []
    hidden_sizes = []
    for mn in model_names:
        config = AutoConfig.from_pretrained("{}".format(mn), trust_remote_code=True)
        logger.info("=== processing model : {}, hidden size: {} ===".format(mn, config.hidden_size))
        hidden_sizes.append(config.hidden_size)

    #model_names.remove("THUDM/chatglm2-6b")
    if args.cons_train:
        reconstructed_dataset, reconstructed_label = get_recons_datasets(
                                                        train_datasets, category_true_labels, num_dps_train, model_names, 
                                                        hidden_sizes, device, args, start_model_id=args.start_model_id,
                                                        mode='train'
                                                    )
        with open('reconstructed_dataset_train.npy', 'wb') as f:
            np.save(f, reconstructed_dataset)
        with open('reconstructed_label_train.npy', 'wb') as f:
            np.save(f, reconstructed_label)

    if args.cons_eval:
        reconstructed_dataset, reconstructed_label = get_recons_datasets(
                                                        eval_datasets, category_true_labels, num_dps_eval, model_names, 
                                                        hidden_sizes, device, args, start_model_id=args.start_model_id,
                                                        mode='eval'
                                                    )
        with open('reconstructed_dataset_eval.npy', 'wb') as f:
            np.save(f, reconstructed_dataset)
        with open('reconstructed_label_eval.npy', 'wb') as f:
            np.save(f, reconstructed_label)

    if args.cons_test:
        reconstructed_dataset, reconstructed_label = get_recons_datasets(
                                                        test_datasets, category_true_labels, num_dps_test, model_names, 
                                                        hidden_sizes, device, args, start_model_id=args.start_model_id,
                                                        mode='test'
                                                    )
        with open('reconstructed_dataset_test.npy', 'wb') as f:
            np.save(f, reconstructed_dataset)
        with open('reconstructed_label_test.npy', 'wb') as f:
            np.save(f, reconstructed_label)
    # model_names.remove("THUDM/chatglm2-6b")
    # logger.info("Calibrated Model names: {}".format(model_names))
    # calibrated_category_true_labels = get_calibrated_true_labels()
    # eval_fuser_acc(test_datasets, calibrated_category_true_labels, model_names)

    # overall_category_acc_collector = {}
    # for i in range(len(model_names)):
    #     category_wise_acc_collector = eval_ind_acc(test_datasets, category_true_labels, model_names, eval_model_id=i)
    #     overall_category_acc_collector[model_names[i]] = category_wise_acc_collector
    # with open("overall_apis_category_acc_collector.json", "w") as j_out:
    #     json.dump(overall_category_acc_collector, j_out)

    logger.info("Fin ..")