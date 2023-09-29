import torch
from torch.cuda.amp import autocast # mixed precision

import logging
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import *
from datasets import load_dataset, Dataset, load_metric # huggingface datasets

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

init_mode = "scratch" # "scratch" or "resume"
eval_mode = "eval_fuser" # "eval_fuser" | "eval_single_model"
eval_model_id = 1 # only for "eval_single_model"
#embedding_dim = 768
embedding_dims = [768, 768, 768, 768]
hidden_dim = 8192
article_batch_size = 16
test_bs = 64

# sample a subset of model for prediction
sample_model = False
num_models_to_sample = 1
API_TO_SAMPLE = 0
sample_model_ids = [API_TO_SAMPLE]

val_split_ratio = 0.6

model_names = ["nickmuchi/finbert-tone-finetuned-fintwitter-classification",
                "joheras/clasificador-poem-sentiment",
                "FinanceInc/auditor_sentiment_finetuned",
                "Kaludi/Reviews-Sentiment-Analysis"
                ]

eval_datasets = {"tfns":load_dataset('zeroshot/twitter-financial-news-sentiment', split="validation"),
                 "poem_sentiment":load_dataset("poem_sentiment", split="validation"),
                 "rsa":Dataset.from_file("./dataset.arrow"),
                 "auditor_sentiment":load_dataset("FinanceInc/auditor_sentiment", split="test"),
                 #"tsa":load_dataset("carblacac/twitter-sentiment-analysis", split="validation")
                }
test_datasets = {"tfns":load_dataset('zeroshot/twitter-financial-news-sentiment', split="validation"),
                "poem_sentiment":load_dataset("poem_sentiment", split="test"),
                "rsa":Dataset.from_file("./dataset.arrow"),
                "auditor_sentiment":load_dataset("FinanceInc/auditor_sentiment", split="test"),
                #"tsa":load_dataset("carblacac/twitter-sentiment-analysis", split="test")
                }

def get_sampled_indices(dataset, data_name="tsa", sampling_ratio=0.1):
    _dataset_ori_size = len(dataset[data_name])
    permuted_indices = np.random.permutation(np.arange(_dataset_ori_size))
    sampled_num = int(_dataset_ori_size * sampling_ratio)
    sampled_indices = permuted_indices[:sampled_num]
    return sampled_indices

#tsa_sampled_indices_val, tsa_sampled_indices_test = get_sampled_indices(eval_datasets, data_name="tsa"), get_sampled_indices(test_datasets, data_name="tsa")

eval_datasets = filter_and_align_sent_labels(eval_datasets, None)
test_datasets = filter_and_align_sent_labels(test_datasets, None)

num_tfns_valid = int(len(eval_datasets["tfns"]["text"]) * val_split_ratio)
num_tsa_valid = int(len(eval_datasets["rsa"]["text"]) * val_split_ratio)
num_cs_valid = int(len(eval_datasets["auditor_sentiment"]["sentence"]) * val_split_ratio)

num_eval_dps = num_tfns_valid + \
                len(eval_datasets["poem_sentiment"]["verse_text"]) + num_tsa_valid \
                + num_cs_valid #+ len(eval_datasets["tsa"]["text"])

num_test_dps = len(eval_datasets["tfns"]["text"]) - num_tfns_valid + \
                len(test_datasets["poem_sentiment"]["verse_text"]) + \
                len(test_datasets['rsa']["text"]) - num_tsa_valid + \
                len(test_datasets['auditor_sentiment']["sentence"]) - num_cs_valid #+ \
                #len(test_datasets["tsa"]["text"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_list = [AutoModelForSequenceClassification.from_pretrained(mn) for mn in model_names]
tokenizer_list = [AutoTokenizer.from_pretrained(mn) for mn in model_names]
max_length_list = [512, 512, 512, 512]

logger.info("&&& Number Eval DataPoints: {}, Number Test DataPoints: {}".format(
                            num_eval_dps, num_test_dps
                        ))

def get_batch(data_item, batch_size, batch_idx):
    return data_item[batch_idx*batch_size:(batch_idx+1)*batch_size]


def get_recons_datasets(datasets, num_dps, num_models, embedding_dims, article_batch_size, valid=True):
    global num_models_to_sample
    if sample_model == True:
        reconstructed_dataset = torch.zeros(num_dps, sum(embedding_dims[:num_models_to_sample]))
    else:
        reconstructed_dataset = torch.zeros(num_dps, sum(embedding_dims))
    reconstructed_label = torch.zeros(num_dps)

    gb_start_idx = 0 # global_batch_counter
    for data_id, (data_name, data_class) in enumerate(datasets.items()):
        logger.info("!!!!!!!!! data id: {}, data name: {}".format(data_id, data_name))
        if data_name == "tfns":
            _len_data = len(data_class["text"])
            slice_idx = int(_len_data * val_split_ratio)
            if valid:
                data_item = data_class["text"][0:slice_idx]
            else:
                data_item = data_class["text"][slice_idx:]
        elif data_name == "poem_sentiment":
            data_item = data_class["verse_text"]
        elif data_name == "rsa":
            _len_data = len(data_class["text"])
            slice_idx = int(_len_data * val_split_ratio)
            if valid:
                data_item = data_class["text"][0:slice_idx]
            else:
                data_item = data_class["text"][slice_idx:]
        elif data_name == "auditor_sentiment":
            _len_data = len(data_class["sentence"])
            slice_idx = int(_len_data * val_split_ratio)
            if valid:
                data_item = data_class["sentence"][0:slice_idx]
            else:
                data_item = data_class["sentence"][slice_idx:]
        elif data_name == "tsa":
            data_item = data_class["text"]
        else:
            raise NotImplementedError("Unsupported Dataset ...")

        if len(data_item)%article_batch_size == 0:
            num_batches = len(data_item)//article_batch_size
        else:
            num_batches = len(data_item)//article_batch_size + 1

        for batch_idx in range(num_batches):
            article_batch = get_batch(data_item=data_item, batch_size=article_batch_size, batch_idx=batch_idx)
            for model_id, (model, tokenizer, max_length) in enumerate(zip(model_list, tokenizer_list, max_length_list)):
                if sample_model and model_id not in sample_model_ids:
                    continue
                model.to(device) # load model to gpu
                inputs = tokenizer(article_batch, 
                                max_length=max_length, 
                                padding=True, truncation=True, return_tensors="pt").to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                    outputs = model(**inputs, output_hidden_states=True)

                #print("** len outputs.hidden_states: {}, outputs.hidden_states[-1] size: {}".format(
                #        len(outputs.hidden_states), outputs.hidden_states[-1].size()
                #    ))
                mean_seq_embedding = torch.mean(outputs.hidden_states[-1], dim=1).data.to("cpu")

                #print("** mean_seq_embedding size: {}, device: {}".format(mean_seq_embedding.size(), mean_seq_embedding.device))
                if sample_model:
                    reconstructed_dataset[gb_start_idx:gb_start_idx+len(article_batch), :] = mean_seq_embedding
                else:
                    reconstructed_dataset[gb_start_idx:gb_start_idx+len(article_batch), model_id*embedding_dims[model_id]:(model_id+1)*embedding_dims[model_id]] \
                                                            = mean_seq_embedding
                
                model.to("cpu") # offload model to cpu

            reconstructed_label[gb_start_idx:gb_start_idx+len(article_batch)] = torch.ones(len(article_batch)) * data_id
            logger.info("---- bengin : {} , end : {} ----".format(gb_start_idx, gb_start_idx+len(article_batch)))
            gb_start_idx += len(article_batch)
            logger.info("# batch_idx: {}/{}, recons_labels: {}".format(batch_idx, num_batches, torch.ones(len(article_batch)) * data_id))
    return reconstructed_dataset, reconstructed_label

if init_mode == "scratch":
    logger.info("*** Processing validation examples ...")
    reconstructed_dataset_valid, reconstructed_label_valid = get_recons_datasets(eval_datasets, 
                                            num_dps=num_eval_dps, num_models=len(model_names), 
                                            embedding_dims=embedding_dims, 
                                            article_batch_size=article_batch_size,
                                            valid=True
                                            )
    logger.info("*** Processing test examples ...")
    reconstructed_dataset_test, reconstructed_label_test = get_recons_datasets(test_datasets, 
                                            num_dps=num_test_dps, num_models=len(model_names), 
                                            embedding_dims=embedding_dims, 
                                            article_batch_size=article_batch_size,
                                            valid=False
                                            )
    reconstructed_label_valid = reconstructed_label_valid.long()
    reconstructed_label_test = reconstructed_label_test.long()
    if sample_model:
        torch.save(reconstructed_dataset_valid, "sa_reconstructed_dataset_valid_sampled.pt")
        torch.save(reconstructed_label_valid, "sa_reconstructed_label_valid_sampled.pt")
        torch.save(reconstructed_dataset_test, "sa_reconstructed_dataset_test_sampled.pt")
        torch.save(reconstructed_label_test, "sa_reconstructed_label_test_sampled.pt")
    else:
        torch.save(reconstructed_dataset_valid, "sa_reconstructed_dataset_valid.pt")
        torch.save(reconstructed_label_valid, "sa_reconstructed_label_valid.pt")
        torch.save(reconstructed_dataset_test, "sa_reconstructed_dataset_test.pt")
        torch.save(reconstructed_label_test, "sa_reconstructed_label_test.pt")
elif init_mode == "resume":
    if sample_model:
        reconstructed_dataset_valid, reconstructed_label_valid = torch.load("sa_reconstructed_dataset_valid_sampled.pt"), torch.load("sa_reconstructed_label_valid_sampled.pt")
        reconstructed_dataset_test, reconstructed_label_test = torch.load("sa_reconstructed_dataset_test_sampled.pt"), torch.load("sa_reconstructed_label_test_sampled.pt")    
    else:
        reconstructed_dataset_valid, reconstructed_label_valid = torch.load("sa_reconstructed_dataset_valid.pt"), torch.load("sa_reconstructed_label_valid.pt")
        reconstructed_dataset_test, reconstructed_label_test = torch.load("sa_reconstructed_dataset_test.pt"), torch.load("sa_reconstructed_label_test.pt")
else:
    raise NotImplementedError("Wrong Init Mode ...")

logger.info("rec dataset size: {}, rec label size: {}".format(
                                reconstructed_dataset_valid.size(), 
                                reconstructed_label_valid.size()
                            ))

# the actual training part
if sample_model:
    classifier = SimpleClassifierForSentimentAnalysis(num_clients=len(model_names), 
                                    embedding_dims=[embedding_dims[API_TO_SAMPLE]], 
                                    hidden_dim=hidden_dim
                                ).to(device)
else:
    classifier = SimpleClassifierForSentimentAnalysis(num_clients=len(model_names), 
                                                    embedding_dims=embedding_dims,
                                                    hidden_dim=hidden_dim
                                                ).to(device)
logger.info("* Arch of the new classifier: {}".format(classifier))
_new_classifier_eps = 10
new_data_loader = torch.utils.data.DataLoader(SimpleDataset(data=reconstructed_dataset_valid.numpy(), 
                                                targets=reconstructed_label_valid.numpy()),
                                                batch_size=64, shuffle=True, num_workers=4)
new_data_loader_test = torch.utils.data.DataLoader(SimpleDataset(data=reconstructed_dataset_test.numpy(), 
                                                targets=reconstructed_label_test.numpy()),
                                                batch_size=test_bs, shuffle=False, num_workers=4)
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

        if batch_idx % 10 == 0:
            logger.info("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                    ep,
                    batch_idx, len(new_data_loader), train_loss/(batch_idx+1), 
                    100.*correct/total, correct, total))
    scheduler.step()

test_loss = 0
correct = 0
total = 0
model_assignment = torch.zeros_like(reconstructed_label_test)
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(new_data_loader_test):
        data, target = data.to(device), target.to(device)

        # the ensemble step
        output = classifier(data)
        smax = F.softmax(output, dim=1)
        pred = smax.argmax(dim=1, keepdim=True)

        model_assignment[batch_idx*test_bs:(batch_idx+1)*test_bs] = pred.flatten()
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        logger.info("target: {}, pred: {}".format(target, pred.flatten()))
    logger.info("@@ Final {}/{}, Accuracy: {:.2f}%".format(
            correct, total, correct/total*100.0))

# evaluation
avg_correct = 0
avg_total = 0

acc_per_data = {}
for data_id, data_name in enumerate(eval_datasets.keys()):
    acc_per_data[data_name] = 0.0

if eval_mode == "eval_fuser":
    assignment_collector = {}
    global_article_id = 0

    for data_id, (data_name, data_class) in enumerate(test_datasets.items()):
        logger.info("!!!!!!!!! data id: {}, data name: {}".format(data_id, data_name))

        if data_name == "tfns":
            _len_data = len(data_class["label"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["text"][slice_idx:]
            label_item = data_class["label"][slice_idx:]
        elif data_name == "poem_sentiment":
            data_item = data_class["verse_text"]
            label_item = data_class["label"]
        elif data_name == "rsa":
            _len_data = len(data_class["target"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["text"][slice_idx:]
            label_item = data_class["target"][slice_idx:]
        elif data_name == "auditor_sentiment":
            _len_data = len(data_class["label"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["sentence"][slice_idx:]
            label_item = data_class["label"][slice_idx:]
        elif data_name == "tsa":
            data_item = data_class["text"]
            label_item = data_class["feeling"]
        else:
            raise NotImplementedError("Unsupported Dataset ...")

        for article_id, (article, label) in enumerate(zip(data_item, label_item)):
            assigned_model_id = int(model_assignment[global_article_id])
            if assigned_model_id not in assignment_collector.keys():
                assignment_collector[assigned_model_id] = {"article": [article], "label": [label], "data_name": [data_name]}
            else:
                assignment_collector[assigned_model_id]["article"].append(article)
                assignment_collector[assigned_model_id]["label"].append(label)
                assignment_collector[assigned_model_id]["data_name"].append(data_name)
            global_article_id += 1

    quality_eval_bs = 8
    for assigned_model, items in assignment_collector.items():
        article_item, label_item, data_name_item = items["article"], items["label"], items["data_name"]
        assert(len(article_item) == len(label_item))
        logger.info("@ Assign model: {}".format(assigned_model))  
        model, tokenizer, max_length = model_list[assigned_model].to(device), tokenizer_list[assigned_model], max_length_list[assigned_model]

        if len(article_item)%quality_eval_bs == 0:
            num_batches = len(article_item)//quality_eval_bs
        else:
            num_batches = len(article_item)//quality_eval_bs + 1

        for batch_idx in range(num_batches):
            logger.info("@ Batch: [{}/{}]".format(batch_idx, num_batches))  
            article_batch = get_batch(data_item=article_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            label_batch = get_batch(data_item=label_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            data_name_batch = get_batch(data_item=data_name_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            label_batch = torch.Tensor(label_batch).to(device)

            inputs = tokenizer(article_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                logits = model(**inputs).logits
            smax = F.softmax(logits, dim=1)
            preds = smax.argmax(dim=1, keepdim=True)

            for dn, pred, label in zip(data_name_batch, preds, label_batch):
                if dn == "auditor_sentiment" and label == 2 and assigned_model != 2:
                    logger.info("** dn: {}, pred: {}, label: {}".format(dn, pred, label))
                    label = 1
                if pred == label:
                    acc_per_data[dn] += 1

            avg_correct += preds.eq(label_batch.view_as(preds)).sum().item()
            avg_total += label_batch.size(0)
        avg_acc = avg_correct * 100.0 / avg_total
        model.to("cpu") # offload model to cpu for memory efficiency

    for data_id, (data_name, data_class) in enumerate(test_datasets.items()):
        if data_name == "tfns":
            _len_data = len(data_class["label"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["text"][slice_idx:]
            label_item = data_class["label"][slice_idx:]
        elif data_name == "poem_sentiment":
            data_item = data_class["verse_text"]
            label_item = data_class["label"]
        elif data_name == "rsa":
            _len_data = len(data_class["target"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["text"][slice_idx:]
            label_item = data_class["target"][slice_idx:]
        elif data_name == "auditor_sentiment":
            _len_data = len(data_class["label"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["sentence"][slice_idx:]
            label_item = data_class["label"][slice_idx:]
        elif data_name == "tsa":
            data_item = data_class["text"]
            label_item = data_class["feeling"]
        else:
            raise NotImplementedError("Unsupported Dataset ...")

        acc_per_data[data_name] /= len(data_item)
elif eval_mode == "eval_single_model":
    global_article_id = 0
    quality_eval_bs = 8
    for data_id, (data_name, data_class) in enumerate(test_datasets.items()):
        if data_name == "tfns":
            _len_data = len(data_class["label"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["text"][slice_idx:]
            label_item = data_class["label"][slice_idx:]
        elif data_name == "poem_sentiment":
            data_item = data_class["verse_text"]
            label_item = data_class["label"]
        elif data_name == "rsa":
            _len_data = len(data_class["target"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["text"][slice_idx:]
            label_item = data_class["target"][slice_idx:]
        elif data_name == "auditor_sentiment":
            _len_data = len(data_class["label"])
            slice_idx = int(_len_data * val_split_ratio)
            data_item = data_class["sentence"][slice_idx:]
            label_item = data_class["label"][slice_idx:]
        elif data_name == "tsa":
            data_item = data_class["text"]
            label_item = data_class["feeling"]
        else:
            raise NotImplementedError("Unsupported Dataset ...")

        model, tokenizer, max_length = model_list[eval_model_id].to(device), tokenizer_list[eval_model_id], max_length_list[eval_model_id]

        if len(data_item) % quality_eval_bs == 0:
            num_batches = len(data_item)//quality_eval_bs
        else:
            num_batches = len(data_item)//quality_eval_bs + 1

        for batch_idx in range(num_batches):
            logger.info("@ Batch: [{}/{}]".format(batch_idx, num_batches))  
            article_batch = get_batch(data_item=data_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            label_batch = get_batch(data_item=label_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            label_batch = torch.Tensor(label_batch).to(device)

            inputs = tokenizer(article_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                logits = model(**inputs).logits
            
            smax = F.softmax(logits, dim=1)
            preds = smax.argmax(dim=1, keepdim=True)

            for pred, label in zip(preds, label_batch):
                #if data_name == "auditor_sentiment" and label == 2:
                #    print("** dn: {}, pred: {}, label: {}".format(data_name, pred, label))
                if data_name == "auditor_sentiment" and label == 2 and eval_model_id != 2:
                    label = 1
                    logger.info("** dn: {}, pred: {}, label: {}".format(data_name, pred, label))
                if pred == label:
                    acc_per_data[data_name] += 1

            avg_correct += preds.eq(label_batch.view_as(preds)).sum().item()
            avg_total += label_batch.size(0)
        avg_acc = avg_correct * 100.0 / avg_total
        model.to("cpu") # offload model to cpu for memory efficiency
        acc_per_data[data_name] /= len(data_item)
else:
    raise NotImplementedError("Unsupported eval mode .")
logger.info("** Final Avg Acc: {} **".format(avg_acc))
logger.info("** Per Data Acc: {} **".format(acc_per_data))