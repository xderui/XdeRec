'''
Description: 
Author: Rigel Ma
Date: 2024-04-21 17:09:04
LastEditors: Rigel Ma
LastEditTime: 2024-04-21 20:21:30
FilePath: loss.py
'''
import torch
import torch.nn.functional as F

def bpr_loss(user_embs, pos_item_embs, neg_item_embs):  
    pos_scores = torch.sum(user_embs * pos_item_embs, dim=-1)
    neg_scores = torch.sum(user_embs * neg_item_embs, dim=-1)
    
    bpr_loss_ = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    return bpr_loss_


def reg_loss(*args):
    return torch.sum(torch.tensor([v.norm(2) for v in args]))


def reg_loss_pow(reg, *args):
    return reg * torch.sum(torch.tensor([v.norm(2).pow(2) for v in args]))


def alignment_loss(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(x, t=2):
    return torch.pdist(F.normalize(x, dim=-1)).pow(2).mul(-t).exp().mean().log()

