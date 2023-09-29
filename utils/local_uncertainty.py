import torch
import torch.nn.functional as F

import numpy as np
to_np = lambda x: x.data.cpu().numpy()

def local_uncertainty(net, net_id, valid_loader, device):
    test_loss = 0
    correct = 0
    total = 0

    smax_scores = []
    ground_truth_labels = []
    with torch.no_grad():
        class_uncertainty = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:0.0, 9:0.0}
        for batch_dix, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            scores = np.max(smax, axis=1)

            smax_scores.extend(scores.tolist())
            ground_truth_labels.extend(target.data.cpu().numpy().tolist())
        for ss, gtl in zip(smax_scores, ground_truth_labels):
            class_uncertainty[gtl] += ss / len(smax_scores)
    print("* Net ID: {}, Class Uncertainty: {}".format(net_id, class_uncertainty))
    return class_uncertainty