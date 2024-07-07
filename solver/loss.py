import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DS_Combin


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, num_coarses):
        super(SoftTargetCrossEntropy, self).__init__()
        self.coarses_criterion = nn.CrossEntropyLoss()

        self.n = len(num_coarses) - 1
        self.num_coarses = num_coarses
        for i in range(1, self.n+1):
            self.num_coarses[i] = self.num_coarses[i] + self.num_coarses[i-1]

    def compute_coarses_loss(self, logits, targets):
        loss = 0
        for i in range(self.n):
            x = logits[i].view(-1, logits[i].size(2))
            y = targets[i+1].view(-1) - self.num_coarses[i]
            loss += self.coarses_criterion(x, y)
        return loss

    def compute_leaves_loss(self, logits, targets):
        loss = torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1)
        loss = loss.mean(dim=0)
        loss = loss.sum()
        return loss

    def forward(self, logits, targets, c_targets):
        loss1 = self.compute_coarses_loss(logits[0], c_targets)
        loss2 = self.compute_leaves_loss(logits[1], targets)
        return loss1 + loss2


class ACELoss(nn.Module):
    def __init__(self, num_classes, num_coarses, criterion, annealing_step=50) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_coarses = num_coarses
        self.criterion = criterion
        self.annealing_step = annealing_step

        self.n = len(num_coarses) - 1
        self.num_coarses = self.num_coarses[:self.n]
        for i in range(1, self.n):
            self.num_coarses[i] = self.num_coarses[i] + self.num_coarses[i-1]

    def compute_kl(self, alpha, num_classes):
        beta = torch.ones((1, alpha.shape[1], num_classes)).cuda()
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=-1, keepdim=True) + lnB + lnB_uni
        return kl

    def compute_loss(self, alpha, targets, num_classes, global_step, prob=False):
        S = torch.sum(alpha, dim=-1, keepdim=True)
        E = alpha - 1
        if not prob:
            targets = F.one_hot(targets, num_classes=num_classes)
        loss = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=True)

        annealing_coef = min(1, global_step / self.annealing_step)

        alp = E * (1 - targets) + 1
        kl_loss = annealing_coef * self.compute_kl(alp, num_classes)

        loss = loss + kl_loss
        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss

    def forward(self, logits, targets, c_targets, global_step, prob=False):
        coarse_loss = torch.tensor(0.).cuda()
        leaves_loss = torch.tensor(0.).cuda()
        for i in range(self.n):
            o = logits[0][i].view(-1, logits[0][i].size(2))
            c = c_targets[i+1] - self.num_coarses[i]
            coarse_loss += self.criterion(o, c.view(-1))

        alpha = logits[1] + 1
        y = targets.unsqueeze(1).repeat(1, 2**(self.n+1), 1)
        leaves_loss += self.compute_loss(alpha, y, self.num_classes, global_step+1, prob)

        alpha = DS_Combin(self.num_classes, alpha.split(1, dim=1))
        alpha = alpha.unsqueeze(1)
        y = targets.unsqueeze(1)
        leaves_loss += self.compute_loss(alpha, y, self.num_classes, global_step+1, prob)

        loss = coarse_loss + leaves_loss
        return loss
