import torch
import argparse
import numpy as np

from utils import *

embed_dims = [4096, 4096, 2560, 4096, 4096, 64, 4096, 4096, 
                4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
test_bs = 10

reconstructed_dataset_val= np.load('reconstructed_dataset_eval.npy')[:, 0:sum(embed_dims[0:1])]
#reconstructed_dataset= np.load('reconstructed_dataset_ckpt_mn3_eval.npy')[:, 10752:10752+4096]
reconstructed_label_val = np.load('reconstructed_label_eval.npy')

reconstructed_dataset_test= np.load('reconstructed_dataset_ckpt_mn0_test.npy')[:, 0:sum(embed_dims[0:1])]
reconstructed_label_test = np.load('reconstructed_label_ckpt_mn0_test.npy')

print("reconstructed val dataset size: {}, reconstructed val label size: {}".format(
        reconstructed_dataset_val.shape, reconstructed_label_val.shape
    ))
print("reconstructed val dataset: {}, reconstructed val label: {}".format(
        reconstructed_label_val, reconstructed_label_val
    ))

print("reconstructed test dataset size: {}, reconstructed test label size: {}".format(
        reconstructed_dataset_test.shape, reconstructed_label_test.shape
    ))
print("reconstructed_dataset: {}, reconstructed_label: {}".format(
        reconstructed_dataset_test, reconstructed_label_test
    ))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = SimpleClassifierForSentimentAnalysis(
               num_clients=16, embedding_dims=embed_dims[0:1], hidden_dim=8192).to(device)
# classifier = LinearForSentimentAnalysis(
#                num_clients=16, embedding_dims=embed_dims[0:4], hidden_dim=8192).to(device)
# classifier = SimpleClassifierForSentimentAnalysis(
#                         num_clients=16, embedding_dims=[4096], hidden_dim=int(4096*2)
#                     ).to(device)

print("* Arch of the new classifier: {}".format(classifier))
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
# optimizer = torch.optim.SGD(classifier.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

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

with open('api_assignment_api1_feat.npy', 'wb') as f:
    np.save(f, api_assignment)