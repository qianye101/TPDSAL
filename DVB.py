
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class DVB(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self):
        super(DVB, self).__init__()
        self.sample_per_class = torch.tensor([1845,46,915,25,121]).cuda()

    def forward(self, input, label, reduction='mean'):
        return dvb(label, input, self.sample_per_class, reduction)


def dvb(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    # spc = sample_per_class.type_as(logits)
    balance = sample_per_class/torch.sum(sample_per_class)
    balance = balance.type_as(logits)
    max_prob = balance.max().item()
    spb = balance.unsqueeze(0).expand(logits.shape[0], -1)
    # prob = balance.unsqueeze(0).expand(logits.shape[0], -1)
    prob = balance / max_prob
    weight = -prob.log() + 1
    logits = logits + spb.log()*spb
    loss = F.cross_entropy(input=logits, target=labels,,weight=weight reduction=reduction)
    return loss
