import torch
from torch.cuda.amp import autocast # mixed precision

import logging
import numpy as np

from transformers import AutoTokenizer, PegasusForConditionalGeneration, FlaxPegasusForConditionalGeneration

from datasets import load_dataset, load_metric # huggingface datasets
from utils import *

from BARTScore.bart_score import BARTScorer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

init_mode = "resume" # "scratch" or "resume"
eval_mode = "eval_fuser" # "eval_fuser" | "eval_single_model"
eval_model_id = 0 # only for "eval_single_model"
embedding_dim = 1024
hidden_dim = 8192
article_batch_size = 32
test_bs = 64

# sample a subset of model for prediction
sample_model = True
num_models_to_sample = 1
sample_model_ids = [i for i in range(num_models_to_sample)]

model_names = ["pegasus-cnn_dailymail", "pegasus-xsum", "pegasus-multi_news", 
                "pegasus-billsum", "pegasus-big_patent", "pegasus-aeslc"]

eval_datasets = {"cnn_dailymail":load_dataset('cnn_dailymail', '3.0.0', split="validation"), # 13.4k
                "xsum":load_dataset('xsum', split="validation"), # 11.3k
                "multi_news":load_dataset('multi_news', split="validation"), # 5.62k
                "billsum":load_dataset('billsum', split="ca_test"), # 1237
                "big_patent":load_dataset("big_patent", "a", split="validation"), # 9674
                "aeslc":load_dataset("aeslc", split="validation")} 
                #"reddit_tifu":load_dataset('reddit_tifu', 'long', split="validation")}
                #"newsroom":load_dataset('newsroom', data_dir="release", split="validation")}
test_datasets = {"cnn_dailymail":load_dataset('cnn_dailymail', '3.0.0', split="test"),
                "xsum":load_dataset('xsum', split="test"),
                "multi_news":load_dataset('multi_news', split="test"), # 5.62k
                "billsum":load_dataset('billsum', split="test"), # 3269
                "big_patent":load_dataset("big_patent", "a", split="test"), #9675
                "aeslc":load_dataset("aeslc", split="test")} 
                #"reddit_tifu":load_dataset('reddit_tifu', 'long', split="test")}
                #"newsroom":load_dataset('newsroom', data_dir="release", split="test")}
num_eval_dps = len(eval_datasets["cnn_dailymail"]["article"]) + len(eval_datasets["xsum"]["document"]) \
                + len(eval_datasets["multi_news"]["document"]) + len(eval_datasets["billsum"]["text"]) \
                + len(eval_datasets["big_patent"]["description"]) + len(eval_datasets["aeslc"]["email_body"])
num_test_dps = len(test_datasets["cnn_dailymail"]["article"]) + len(test_datasets["xsum"]["document"]) \
                + len(test_datasets["multi_news"]["document"]) + len(test_datasets["billsum"]["text"]) \
                + len(test_datasets["big_patent"]["description"]) + len(eval_datasets["aeslc"]["subject_line"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_list = [PegasusForConditionalGeneration.from_pretrained("google/{}".format(mn)) for mn in model_names]
tokenizer_list = [AutoTokenizer.from_pretrained("google/{}".format(mn)) for mn in model_names]
max_length_list = [1024, 512, 1024, 1024, 1024, 512]

logger.info("&&& Number Eval DataPoints: {}, Number Test DataPoints: {}".format(
                            num_eval_dps, num_test_dps
                        ))

def get_batch(data_item, batch_size, batch_idx):
    return data_item[batch_idx*batch_size:(batch_idx+1)*batch_size]

def get_recons_datasets(datasets, num_dps, num_models, embedding_dim, article_batch_size):
    if sample_model == True:
        reconstructed_dataset = torch.zeros(num_dps, embedding_dim)
    else:
        reconstructed_dataset = torch.zeros(num_dps, embedding_dim*num_models)
    reconstructed_label = torch.zeros(num_dps)

    gb_start_idx = 0 # global_batch_counter
    for data_id, (data_name, data_class) in enumerate(datasets.items()):
        if data_name == "cnn_dailymail":
            data_item = data_class["article"]
        elif data_name == "xsum":
            data_item = data_class["document"]
        elif data_name == "multi_news":
            data_item = data_class["document"]
        elif data_name == "billsum":
            data_item = data_class["text"]
        elif data_name == "big_patent":
            data_item = data_class["description"]
        elif data_name == "aeslc":
            data_item = data_class["email_body"]
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
                    max_length=max_length, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                    summary = model.generate(inputs["input_ids"], num_beams=1, do_sample=False,
                                return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

                seq_embedding = torch.stack([dhs[-1].view(len(article_batch),-1) for dhs in summary.decoder_hidden_states])
                mean_seq_embedding = torch.mean(seq_embedding, dim=0)

                if sample_model:
                    reconstructed_dataset[gb_start_idx:gb_start_idx+len(article_batch), :] = mean_seq_embedding
                else:
                    reconstructed_dataset[gb_start_idx:gb_start_idx+len(article_batch), model_id*embedding_dim:(model_id+1)*embedding_dim] = mean_seq_embedding
                
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
                                            embedding_dim=embedding_dim, article_batch_size=article_batch_size)
    logger.info("*** Processing test examples ...")
    reconstructed_dataset_test, reconstructed_label_test = get_recons_datasets(test_datasets, 
                                            num_dps=num_test_dps, num_models=len(model_names), 
                                            embedding_dim=embedding_dim, article_batch_size=article_batch_size)
    reconstructed_label_valid = reconstructed_label_valid.long()
    reconstructed_label_test = reconstructed_label_test.long()
    if sample_model:
        torch.save(reconstructed_dataset_valid, "reconstructed_dataset_valid_sampled.pt")
        torch.save(reconstructed_label_valid, "reconstructed_label_valid_sampled.pt")
        torch.save(reconstructed_dataset_test, "reconstructed_dataset_test_sampled.pt")
        torch.save(reconstructed_label_test, "reconstructed_label_test_sampled.pt")
    else:
        torch.save(reconstructed_dataset_valid, "reconstructed_dataset_valid.pt")
        torch.save(reconstructed_label_valid, "reconstructed_label_valid.pt")
        torch.save(reconstructed_dataset_test, "reconstructed_dataset_test.pt")
        torch.save(reconstructed_label_test, "reconstructed_label_test.pt")
elif init_mode == "resume":
    if sample_model:
        reconstructed_dataset_valid, reconstructed_label_valid = torch.load("reconstructed_dataset_valid_sampled.pt"), torch.load("reconstructed_label_valid_sampled.pt")
        reconstructed_dataset_test, reconstructed_label_test = torch.load("reconstructed_dataset_test_sampled.pt"), torch.load("reconstructed_label_test_sampled.pt")    
    else:
        reconstructed_dataset_valid, reconstructed_label_valid = torch.load("reconstructed_dataset_valid.pt"), torch.load("reconstructed_label_valid.pt")
        reconstructed_dataset_test, reconstructed_label_test = torch.load("reconstructed_dataset_test.pt"), torch.load("reconstructed_label_test.pt")
else:
    raise NotImplementedError("Wrong Init Mode ...")

logger.info("rec dataset size: {}, rec label size: {}".format(
                                reconstructed_dataset_valid.size(), 
                                reconstructed_dataset_valid.size()
                            ))

# the actual training part
if sample_model:
    classifier = SimpleClassifier(num_clients=len(model_names), 
                                embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                num_sampled_clients=len(sample_model_ids)).to(device)
else:
    classifier = SimpleClassifier(num_clients=len(model_names), embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
logger.info("* Arch of the new classifier: {}".format(classifier))
_new_classifier_eps = 5
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

    logger.info("@@ Final {}/{}, Accuracy: {:.2f}%".format(
            correct, total, correct/total*100.0))

# evaluation
avg_rouge2_score = 0.0
avg_bart_score = 0.0

rouge2_score_per_data = {}
bart_score_per_data = {}
for data_id, data_name in enumerate(eval_datasets.keys()):
    rouge2_score_per_data[data_name], bart_score_per_data[data_name] = 0.0, 0.0

rouge = load_metric('rouge')
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

if eval_mode == "eval_fuser":
    assignment_collector = {}
    global_article_id = 0

    for data_id, (data_name, data_class) in enumerate(test_datasets.items()):
        if data_name == "cnn_dailymail":
            data_item = data_class["article"]
            reference_item = data_class["highlights"]
        elif data_name == "xsum":
            data_item = data_class["document"]
            reference_item = data_class["summary"]
        elif data_name == "multi_news":
            data_item = data_class["document"]
            reference_item = data_class["summary"]
        elif data_name == "billsum":
            data_item = data_class["text"]
            reference_item = data_class["summary"]
        elif data_name == "big_patent":
            data_item = data_class["description"]
            reference_item = data_class["abstract"]
        elif data_name == "aeslc":
            data_item = data_class["email_body"]
            reference_item = data_class["subject_line"]
        else:
            raise NotImplementedError("Unsupported Dataset ...")

        for article_id, (article, reference) in enumerate(zip(data_item, reference_item)):
            assigned_model_id = int(model_assignment[global_article_id])
            if assigned_model_id not in assignment_collector.keys():
                assignment_collector[assigned_model_id] = {"article": [article], "reference": [reference], "data_name": [data_name]}
            else:
                assignment_collector[assigned_model_id]["article"].append(article)
                assignment_collector[assigned_model_id]["reference"].append(reference)
                assignment_collector[assigned_model_id]["data_name"].append(data_name)
            global_article_id += 1

    quality_eval_bs = 16
    for assigned_model, items in assignment_collector.items():
        article_item, reference_item, data_name_item = items["article"], items["reference"], items["data_name"]
        assert(len(article_item) == len(reference_item))
        logger.info("@ Assign model: {}".format(assigned_model))  
        model, tokenizer, max_length = model_list[assigned_model].to(device), tokenizer_list[assigned_model], max_length_list[assigned_model]

        if len(article_item)%quality_eval_bs == 0:
            num_batches = len(article_item)//quality_eval_bs
        else:
            num_batches = len(article_item)//quality_eval_bs + 1

        for batch_idx in range(num_batches):
            logger.info("@ Batch: [{}/{}]".format(batch_idx, num_batches))  
            article_batch = get_batch(data_item=article_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            reference_batch = get_batch(data_item=reference_item, batch_size=quality_eval_bs, batch_idx=batch_idx)
            data_name_batch = get_batch(data_item=data_name_item, batch_size=quality_eval_bs, batch_idx=batch_idx)

            inputs = tokenizer(article_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                summary_ids = model.generate(inputs["input_ids"])
            
            summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            article_rouge2_score_temp = rouge.compute(
                                        predictions=summary, 
                                        references=reference_batch,
                                        use_aggregator=False
                                    )
            for dn, ar2s in zip(data_name_batch, article_rouge2_score_temp['rouge2']):
                rouge2_score_per_data[dn] += (ar2s.fmeasure * 100.0)

            article_rouge2_score = np.sum([ar2s.fmeasure * 100.0 for ar2s in article_rouge2_score_temp['rouge2']])
            article_bart_score_temp = bart_scorer.score(
                            summary,
                            reference_batch,
                            batch_size=4
                        )
            assert len(data_name_batch) == len(article_bart_score_temp)

            for dn, abarts in zip(data_name_batch, article_bart_score_temp):
                bart_score_per_data[dn] += abarts
            article_bart_score = np.sum(article_bart_score_temp)

            avg_rouge2_score += article_rouge2_score
            avg_bart_score += article_bart_score
        model.to("cpu") # offload model to cpu for memory efficiency

    for data_id, (data_name, data_class) in enumerate(test_datasets.items()):
        if data_name == "cnn_dailymail":
            data_item = data_class["article"]
            reference_item = data_class["highlights"]
        elif data_name == "xsum":
            data_item = data_class["document"]
            reference_item = data_class["summary"]
        elif data_name == "multi_news":
            data_item = data_class["document"]
            reference_item = data_class["summary"]
        elif data_name == "billsum":
            data_item = data_class["text"]
            reference_item = data_class["summary"]
        elif data_name == "big_patent":
            data_item = data_class["description"]
            reference_item = data_class["abstract"]
        elif data_name == "aeslc":
            data_item = data_class["email_body"]
            reference_item = data_class["subject_line"]
        else:
            raise NotImplementedError("Unsupported Dataset ...")
        rouge2_score_per_data[data_name] /= len(data_item)
        bart_score_per_data[data_name] /= len(data_item)
    avg_rouge2_score /= num_test_dps 
    avg_bart_score /= num_test_dps
elif eval_mode == "eval_single_model":
    global_article_id = 0
    quality_eval_bs = 16
    for data_id, (data_name, data_class) in enumerate(test_datasets.items()):
        if data_name == "cnn_dailymail":
            data_item = data_class["article"]
            reference_item = data_class["highlights"]
        elif data_name == "xsum":
            data_item = data_class["document"]
            reference_item = data_class["summary"]
        elif data_name == "multi_news":
            data_item = data_class["document"]
            reference_item = data_class["summary"]
        elif data_name == "billsum":
            data_item = data_class["text"]
            reference_item = data_class["summary"]
        elif data_name == "big_patent":
            data_item = data_class["description"]
            reference_item = data_class["abstract"]
        elif data_name == "aeslc":
            data_item = data_class["email_body"]
            reference_item = data_class["subject_line"]
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
            reference_batch = get_batch(data_item=reference_item, batch_size=quality_eval_bs, batch_idx=batch_idx)

            inputs = tokenizer(article_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # fp16
                summary_ids = model.generate(inputs["input_ids"])
            
            summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            article_rouge2_score_temp = rouge.compute(
                                        predictions=summary, 
                                        references=reference_batch,
                                        use_aggregator=False
                                    )
            article_rouge2_score = np.sum([ar2s.fmeasure * 100.0 for ar2s in article_rouge2_score_temp['rouge2']])
            article_bart_score = np.sum(bart_scorer.score(
                            summary,
                            reference_batch,
                            batch_size=4
                        ))
            avg_rouge2_score += article_rouge2_score
            avg_bart_score += article_bart_score
            rouge2_score_per_data[data_name] += (article_rouge2_score/len(data_item))
            bart_score_per_data[data_name] += (article_bart_score/len(data_item))

    avg_rouge2_score /= num_test_dps 
    avg_bart_score /= num_test_dps
else:
    raise NotImplementedError("Unsupported eval mode .")
logger.info("** Final Avg ROUGE2 Score: {}, Avg BART Score: {} **".format(avg_rouge2_score, avg_bart_score))
logger.info("** Per Data ROUGE2 Score: {}, Per Data BART Score: {} **".format(rouge2_score_per_data, bart_score_per_data))