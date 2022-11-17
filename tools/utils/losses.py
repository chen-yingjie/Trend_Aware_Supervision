import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import stats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RegressionLoss(nn.Module):

    def __init__(self,
                 weights=None,
                 loss_lambda=1,
                 use_weight=False,
                 **kwargs):
        super(RegressionLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.use_weight = use_weight
        if use_weight:
            self.weights = torch.tensor(weights).to(device)
       
    def forward(self, outputs, targets, au_class, num_seg, rand_idx=None, **kwargs):
        seg_len = outputs.shape[0] // num_seg
        selected_idx = sorted(
            sum([[i * seg_len, (i + 1) * seg_len - 1]
                for i in range(num_seg)], []))
        
        output = outputs[selected_idx]
        target = targets[selected_idx]
        au_class_ = au_class[selected_idx]

        target_ = target.data.cpu().numpy()
        au_class_ = au_class_.data.cpu().numpy()

        N, num_class = output.size()
        if self.use_weight:
            weights = torch.zeros((target_.shape[0]), device=device)
            for i in range(N):
                c = int(au_class_[i])
                weights[i] = 1 / self.weights[c, int(target_[i] * 5)]
        else:
            weights = torch.ones((target_.shape[0]), device=device)

        loss = torch.sum(
            torch.mul(weights, torch.mul(output - target,
                                            output - target)))

        return self.loss_lambda * loss / output.shape[0]


class SubjectVarianceLoss(nn.Module):

    def __init__(self,
                 num_class,
                 weights=None,
                 loss_lambda=1,
                 use_weight=False,
                 start_epoch=0,
                 key_only=True,
                 **kwargs):
        super(SubjectVarianceLoss, self).__init__()
        self.num_class = num_class
        self.loss_lambda = loss_lambda
        self.use_weight = use_weight
        if use_weight:
            self.weights = torch.tensor(weights).to(device)

        self.start_epoch = start_epoch

        self.key_only = key_only
    
    def forward(self, outputs, targets, au_class, num_seg, epoch, neutral_feature=None, short_mark=None, **kwargs):
        
        loss = torch.tensor(0.).to(device)

        if epoch < self.start_epoch:
            return loss

        seg_len = outputs.shape[0] // num_seg

        if self.key_only:
            selected_idx = sorted(
                sum([[i * seg_len, (i + 1) * seg_len - 1]
                        for i in range(num_seg)], []))
            outputs = outputs[selected_idx]
            targets = targets[selected_idx]
            au_class = au_class[selected_idx]
            if neutral_feature is not None:
                neutral_feature = neutral_feature[selected_idx]
            seg_len = 2
        
        targets_ = targets.cpu().detach().numpy()
        for i in range(5, -1, -1):
            idx = -0.1 + 0.2 * i
            targets_[(targets_ >= idx) * (targets_ < idx + 0.2)] = int(i)
        targets_ = np.array(targets_, dtype=np.int8)

        for c in range(self.num_class):
            label_dict = {k: [] for k in range(6)}
            for seg_idx in range(num_seg):
                if au_class[seg_idx * seg_len] != c:
                    pass
                if neutral_feature is not None:
                    output = outputs[seg_idx * seg_len:(seg_idx + 1) * seg_len, c] - neutral_feature[seg_idx, c]
                else:
                    output = outputs[seg_idx * seg_len:(seg_idx + 1) * seg_len, c]
                for i in range(output.shape[0]):
                    label_dict[targets_[seg_idx * seg_len][0]].append(output[i])

            loss_tmp = torch.tensor(0.).to(device)
            for k, v in label_dict.items():
                if len(v):
                    v_ = torch.stack(v)
                    mask = (torch.ones(v_.shape[0]) - torch.eye(v_.shape[0])).to(device)
                    sim = F.cosine_similarity(v_[None, :, :], v_[:, None, :], dim=-1)
                    sim = 1 - sim
                    loss_tmp += (sim * mask).mean()
               
            loss += loss_tmp

        return self.loss_lambda * loss / self.num_class


class MonotonicTrendLoss(nn.Module):

    def __init__(self, dist='l2', loss_lambda=1, **kwargs):
        super(MonotonicTrendLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.dist = dist

    def forward(self, outputs, targets, au_class, num_seg, short_mark=None, **kwargs):
        loss = 0.
        seg_len = outputs.shape[0] // num_seg

        seg_mask = (torch.tril(torch.ones(seg_len - 1, seg_len), diagonal=0) -
                    torch.tril(torch.ones(seg_len - 1, seg_len), diagonal=1) +
                    torch.eye(seg_len - 1, seg_len)).to(outputs.device)

        for seg_idx in range(num_seg):
            output = outputs[seg_idx * seg_len:(seg_idx + 1) * seg_len]
            target = targets[seg_idx * seg_len:(seg_idx + 1) * seg_len]

            if output.shape[-1] > 1:
                base_ascend = output[0].repeat(seg_len, 1).detach()
                base_descend = output[-1].repeat(seg_len, 1).detach()
                if self.dist == 'l2':
                    dist_ascend = torch.sum((output - base_ascend)**2,
                                            dim=1).unsqueeze(-1)
                else:
                    dist_ascend = 1 - torch.mul(output, base_ascend)

                loss += torch.sum(
                    torch.max(
                        torch.mm(seg_mask, output),
                        torch.zeros_like(torch.mm(seg_mask,
                                                  dist_ascend)).to(device)))
            else:
                loss += torch.sum(
                    torch.max(
                        torch.mm(seg_mask, output),
                        torch.zeros_like(torch.mm(seg_mask,
                                                  output)).to(device)))

        return self.loss_lambda * loss / outputs.shape[0]


class VicinalDistributionLoss(nn.Module):

    def __init__(self, size_average=True, loss_lambda=1, **kwargs):
        super(VicinalDistributionLoss, self).__init__()
        self.size_average = size_average
        self.loss_lambda = loss_lambda

    def forward(self, outputs, targets, **kwargs):
        if len(outputs.size()) > 2:
            outputs = F.softmax(outputs.permute(0, 2, 1),
                                dim=-1).argmax(-1) / 5.0

        num_class = outputs.size()[-1]
        outputs = outputs.view(-1, num_class)
        targets = targets.view(-1, num_class).requires_grad_(False)

        N, num_class = outputs.size()
        loss_buff = torch.sum(torch.mul(outputs - targets, outputs - targets))

        return self.loss_lambda * loss_buff / (num_class * N)


if __name__ == '__main__':
    targets = [[1, 0, 0, 1, 1, 0, 1]]
    outputs = [[0.5, 0.4, 0.1, 0.6, 0.7, 0.2, 0.3]]

    targets = torch.tensor(targets, dtype=torch.float)
    outputs = torch.tensor(outputs, dtype=torch.float)