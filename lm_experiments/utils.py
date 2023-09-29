import os
import random
import numpy as np
import logging

from torch import nn
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

MMLU_AVAIL_CATEGORIES = [
'high_school_european_history', 'business_ethics', 'clinical_knowledge', 
'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 
'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 
'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 
'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 
'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 
'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 
'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 
'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 
'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 
'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 
'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology'
]

class SimpleClassifier(nn.Module):
    def __init__(self, num_clients, embedding_dim, hidden_dim=1024, num_sampled_clients=None):
        super(SimpleClassifier, self).__init__()
        if num_sampled_clients is None:
            num_sampled_clients = num_clients

        self.fc1 = nn.Linear(int(num_sampled_clients*embedding_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_clients)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SimpleClassifierForSentimentAnalysis(nn.Module):
    def __init__(self, num_clients, embedding_dims, hidden_dim=1024):
        super(SimpleClassifierForSentimentAnalysis, self).__init__()

        self.fc1 = nn.Linear(sum(embedding_dims), hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_clients)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class LinearForSentimentAnalysis(nn.Module):
    def __init__(self, num_clients, embedding_dims, hidden_dim=1024):
        super(LinearForSentimentAnalysis, self).__init__()
        self.weight = nn.Linear(sum(embedding_dims), num_clients)

    def forward(self, x):
        x = self.weight(x)
        return x

class SimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, targets, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img, target = self.data[idx, :], self.targets[idx]
        return img, target


def filter_and_align_sent_labels(datasets, tsa_sample_indices):
    new_datasets = {"tfns":{}, "poem_sentiment":{}, "auditor_sentiment":{}, "rsa":{}}
    for data_id, (data_name, data_class) in enumerate(datasets.items()):
        new_data = []
        new_labels = []
        if data_name == "tfns":
            # "LABEL_0": "Bearish" (negative), "LABEL_1": "Bullish" (positive), "LABEL_2": "Neutral"
            for item_idx, (data, label) in enumerate(zip(data_class["text"], data_class["label"])):
                if label == 2:
                    continue
                else:
                    new_data.append(data)
                    new_labels.append(label)
            new_datasets["tfns"]["text"] = new_data
            new_datasets["tfns"]["label"] = new_labels
        elif data_name == "poem_sentiment":
            # 0 = negative; 1 = positive; 2 = no impact 3 = mixed (both negative and positive)
            for item_idx, (data, label) in enumerate(zip(data_class["verse_text"], data_class["label"])):
                if label in (2, 3):
                    continue
                else:
                    new_data.append(data)
                    new_labels.append(label)
            new_datasets["poem_sentiment"]["verse_text"] = new_data
            new_datasets["poem_sentiment"]["label"] = new_labels            
        elif data_name == "rsa":
            # 0 = 'Negative', 1 = 'Positive'
            for item_idx, (data, label) in enumerate(zip(data_class["text"], data_class["target"])):
                #if item_idx in rsa_sample_indices:
                new_data.append(data)
                new_labels.append(label)
            new_datasets["rsa"]["text"] = new_data
            new_datasets["rsa"]["target"] = new_labels
        elif data_name == "auditor_sentiment":
            # 'negative' - (0); 'neutral' - (1); 'positive' - (2)
            for item_idx, (data, label) in enumerate(zip(data_class["sentence"], data_class["label"])):
                if label == 1:
                    continue
                #elif label == 2:
                #    new_data.append(data)
                #    new_labels.append(1)
                else:
                    new_data.append(data)
                    new_labels.append(label)                    
            new_datasets["auditor_sentiment"]["sentence"] = new_data
            new_datasets["auditor_sentiment"]["label"] = new_labels  
        elif data_name == "tsa":
            # down sample the rsa data a bit, otherwise it's too large
            # permuted_indices = np.random.permutation(np.arange(len(data_class["text"])))
            # sampled_num = int(len(data_class["text"]) * 0.2)
            # sampled_indices = permuted_indices[:sampled_num]

            # 0 = 'Negative', 1 = 'Positive'
            for item_idx, (data, label) in enumerate(zip(data_class["text"], data_class["feeling"])):
                if item_idx in tsa_sample_indices:
                    new_data.append(data)
                    new_labels.append(label)
            new_datasets["tsa"]["text"] = new_data
            new_datasets["tsa"]["feeling"] = new_labels  
        else:
            raise NotImplementedError("Unsupported Dataset ...")
    del datasets
    return new_datasets

to_np = lambda x: x.data.cpu().numpy()

def get_logger():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

def seed(seed):
    logger = get_logger()
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Seeded everything")