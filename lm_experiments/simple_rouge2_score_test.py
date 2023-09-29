import torch
from torch.cuda.amp import autocast # mixed precision

import numpy as np

import jax.numpy as jnp
from transformers import AutoTokenizer, PegasusForConditionalGeneration, FlaxPegasusForConditionalGeneration

from datasets import load_dataset, load_metric # huggingface datasets
from utils import *

from BARTScore.bart_score import BARTScorer

embedding_dim = 1024
hidden_dim = 8192
article_batch_size = 32
test_bs = 64

model_names = ["pegasus-cnn_dailymail"]

data = load_dataset('cnn_dailymail', '3.0.0', split="test")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_list = [PegasusForConditionalGeneration.from_pretrained("google/{}".format(mn)) for mn in model_names]
tokenizer_list = [AutoTokenizer.from_pretrained("google/{}".format(mn)) for mn in model_names]
max_length_list = [1024, 512, 1024, 1024, 1024]


rouge = load_metric('rouge')


data_item = data["article"]
reference_item = data["highlights"]
model, tokenizer, max_length = model_list[0].to(device), tokenizer_list[0], max_length_list[0]
for article_id, (article, reference) in enumerate(zip(data_item, reference_item)):
    inputs = tokenizer(article, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)

    summary_ids = model.generate(inputs["input_ids"])
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    article_rouge2_score = rouge.compute(
                                    predictions=summary, 
                                    references=[reference]
                                )['rouge2'].mid.fmeasure

    print("* Article ID: {} ROUGE score: {}".format(article_id, article_rouge2_score))