import json

from transformers import AutoTokenizer, AutoModel
import datasets

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from BARTScore.bart_score import BARTScorer

import numpy as np
  

datum = []
file_names = ['luminous_supreme_ss.json', 'j1_large_ss.json', 'davinci_003_ss.json']
embedding_dim = 384
hidden_dim = 2048

num_trials = 3
score_type = 'bart-score' # 'bart-score'|'rouge-2'

rouge2_threshold = 0.2
num_valid = int(534 * num_trials)
num_test = int(466 * num_trials)

multi_class_label = False #True

if score_type == "rouge-2":
    rouge = datasets.load_metric('rouge')
elif score_type == "bart-score":
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
else:
    raise NotImplementedError("Unsupported Score type ...")


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# Opening JSON file
for fi in file_names:
    with open(fi) as f:
        # returns JSON object as 
        # a dictionary
        datum.append(json.load(f)["request_states"])


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SimpleClassifier(nn.Module):
    def __init__(self, num_clients, embedding_dim, hidden_dim=1024):
        super(SimpleClassifier, self).__init__()

        self.fc1 = nn.Linear(int(num_clients*embedding_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_clients)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
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


constructed_feat_valid = torch.zeros(num_valid, 384*(len(datum)+1))
constructed_feat_test = torch.zeros(num_test, 384*(len(datum)+1))

if multi_class_label:
    constcuted_label_valid = torch.zeros(num_valid, len(datum))
    constcuted_label_test = torch.zeros(num_test, len(datum))
else:
    constcuted_label_valid = torch.zeros(num_valid)
    constcuted_label_test = torch.zeros(num_test)


val_req_count = 0
test_req_count = 0
for req_id, req in enumerate(zip(*(datum[i] for i in range(len(datum))))):
    references = [req[0]["instance"]["references"][0]["output"]["text"]]

    # Sentences we want sentence embeddings for
    if score_type == "rouge-2":
        rouge2_results = [rouge.compute(
                            predictions=[req[i]["result"]["completions"][0]["text"]], 
                            references=references
                        )['rouge2'].mid.fmeasure for i in range(len(datum))]
        score_res = rouge2_results
    elif score_type == "bart-score":
        bart_score_results = [bart_scorer.score(
                            [req[i]["result"]["completions"][0]["text"]], 
                            references,
                            batch_size=4
                        )[0] for i in range(len(datum))]
        score_res = bart_score_results
    else:
        raise NotImplementedError("Unsupported Score type ...")
    if multi_class_label:
        if score_type == "rouge-2":
            rouge2_results_multi_label = torch.where(torch.Tensor(rouge2_results) > rouge2_threshold, 1.0, 0.0)
            score_res_multilabel = rouge2_results_multi_label
        elif score_type == "bart-score":
            raise NotImplementedError("multi label setting for bart score has not been implemented yet. ")
        else:
            raise NotImplementedError("Unsupported Score type ...")
        print("@ req_id: {}, rouge2_results: {}, {}_results_multi_label: {}".format(
                            req_id, score_res, score_type, score_res_multilabel
                        ))
    else:
        print("@ req_id: {}, {}_results: {}".format(req_id, score_type, score_res))
    sentences = [req[0]["instance"]["input"]["text"]] + [req[i]["result"]["completions"][0]["text"] for i in range(len(datum))]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    if req[0]["instance"]["split"] == "test":
        constructed_feat_test[test_req_count, :] = sentence_embeddings.flatten()
        if multi_class_label:
            constcuted_label_test[test_req_count, :] = score_res_multilabel
        else:
            l = np.argmax(score_res)
            constcuted_label_test[test_req_count] = l
        test_req_count += 1
    else:
        constructed_feat_valid[val_req_count, :] = sentence_embeddings.flatten()
        if multi_class_label:
            constcuted_label_valid[val_req_count, :] = score_res_multilabel
        else:
            l = np.argmax(score_res)
            constcuted_label_valid[val_req_count] = l
        val_req_count += 1     

print(constcuted_label_test)
print(constcuted_label_valid)

# the actual training part
classifier = SimpleClassifier(num_clients=len(datum)+1, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
print("* Arch of the new classifier: {}".format(classifier))
_new_classifier_eps = 60
new_data_loader = torch.utils.data.DataLoader(SimpleDataset(data=constructed_feat_valid.numpy(), targets=constcuted_label_valid.numpy()),
                                                batch_size=64, shuffle=True, num_workers=4)
new_data_loader_test = torch.utils.data.DataLoader(SimpleDataset(data=constructed_feat_test.numpy(), targets=constcuted_label_test.numpy()),
                                                batch_size=64, shuffle=True, num_workers=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_new_classifier_eps)

classifier.train()
train_loss = 0
correct = 0
total = 0
for ep in range(_new_classifier_eps):
    for batch_idx, (inputs, targets) in enumerate(new_data_loader):
        if multi_class_label:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
        else:
            inputs, targets = inputs.float().to(device), targets.long().to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if multi_class_label:
            smax = F.softmax(outputs, dim=1)
            predicted = torch.where(smax > 0.5, 1.0, 0.0)
        else:
            _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print("Ep: {}, Training: {}/{}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                    ep,
                    batch_idx, len(new_data_loader), train_loss/(batch_idx+1), 
                    100.*correct/total, correct, total))
    scheduler.step()

test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch_dix, (data, target) in enumerate(new_data_loader_test):
        data, target = data.to(device), target.to(device)

        # the ensemble step
        output = classifier(data)

        smax = F.softmax(output, dim=1)

        if multi_class_label:
            pred = torch.where(smax > 0.5, 1.0, 0.0)
            print("** predicted: {}, targets: {}".format(pred, target))
            correct += (pred.eq(target.view_as(pred)).sum().item()/len(datum))
        else:
            pred = smax.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    print("@@ Final {}/{}, Accuracy: {:.2f}%".format(
            correct, total, correct/total*100.0))