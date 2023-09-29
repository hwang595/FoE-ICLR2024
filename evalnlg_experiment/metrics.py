import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import traceback
from bert_score import BERTScorer
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from typing import List
from evaluate import load
import nltk
import pandas as pd
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
nltk.download('punkt')

class UniEvalScorerOurs:
    def __init__(self, task = 'summarization', device='cuda'):
        self.task=task
        self.device=device
  
    def score(self, srcs, hyps, refs, agg="mean"):
        
        assert len(hyps)==len(refs) 
        
        evaluator = get_evaluator(self.task, device=self.device)

        data = convert_to_json(output_list=hyps, src_list=srcs) 
        eval_scores1 = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency'], overall=False, print_result=False)

        refsT = np.array(refs).T.tolist()
        eval_scores2=[]
        for j in range(np.array(refs).shape[1]):
            data = convert_to_json(output_list=hyps[:], src_list=srcs[:], ref_list=refsT[j][:])
            evalu = evaluator.evaluate(data, dims=['relevance'], overall=False, print_result=False)
            eval_scores2.append(np.array(pd.DataFrame(evalu)).squeeze().tolist())

        if agg=='max':
            return np.hstack((np.array(pd.DataFrame(eval_scores1)),
                          np.array(eval_scores2).mean(axis=0).reshape((-1,1)))).tolist()
        elif agg=='mean':
            return np.hstack((np.array(pd.DataFrame(eval_scores1)),
                          np.array(eval_scores2).max(axis=0).reshape((-1,1)))).tolist()
        

class BERTScorerOurs:
    def __init__(self, device='cuda', model_type='microsoft/deberta-large-mnli'):
        self.device=device
        self.model_type=model_type
  
    def score(self, hyps, refs, agg="mean", batch_size=16):
        
        assert len(hyps)==len(refs) 
        
        bert_scorer = BERTScorer(device=self.device, model_type=self.model_type)
        refsT = np.array(refs).T.tolist()
        scores = torch.stack([torch.stack(bert_scorer.score(hyps, r, batch_size=batch_size)) for r in refsT])
        scores = scores.numpy()
        
        if agg=='max':
            scores = scores.max(axis=0).T.tolist()
        elif agg=='mean':
            scores = scores.mean(axis=0).T.tolist()
        
        return scores
    

# code from https://github.com/neulab/BARTScore
class PegasusScorer:
    def __init__(self, device='cuda', max_length=1024, checkpoint='google/pegasus-cnn_dailymail'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.model = PegasusForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        
        # Tokenizer
        self.tokenizer = PegasusTokenizer.from_pretrained(checkpoint)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/pegasus.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))

        
class BARTScorer:
    def __init__(self, device='cuda', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))