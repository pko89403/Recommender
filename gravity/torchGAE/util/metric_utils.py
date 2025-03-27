import os 

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from util.deco_util import timeit


@timeit
def get_roc_score(emb, adj_orig, edges_pos, edges_neg, device, lamb=5.0):    
    emb_sp =emb.detach()
    emb_t_sp = torch.transpose(emb, 0, 1).detach()
    
    # Predict 
    adj_rec = torch.mm(emb_sp, emb_t_sp)
    preds = [] 
    pos = []
        
    for e in edges_pos:
        preds.append(torch.sigmoid(adj_rec[e[0], e[1]]).detach().cpu().numpy())
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = [] 
    neg = [] 
    for e in edges_neg:
        preds_neg.append(torch.sigmoid(adj_rec[e[0], e[1]]).detach().cpu().numpy())
        neg.append(adj_orig[e[0], e[1]])
        
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score

@timeit
def get_roc_score_split(emb, adj_orig, edges_pos, edges_neg, device):
    adj_rec = None 
    emb = emb.detach().to(device)
    emb_t = torch.transpose(emb, 0, 1)

    emb_splits = torch.split(emb, 100, dim=0)    
    for idx, emb_split in enumerate(emb_splits):
        tmp_adj_rec = torch.mm(emb_split, emb_t)
        if idx == 0:
            adj_rec = tmp_adj_rec
        else:        
            adj_rec = torch.cat((adj_rec, tmp_adj_rec), dim=0)

    preds = [] 
    pos = []
        
    for e in edges_pos:
        preds.append(torch.sigmoid(adj_rec[e[0], e[1]]).detach().cpu().numpy())
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = [] 
    neg = [] 
    for e in edges_neg:
        preds_neg.append(torch.sigmoid(adj_rec[e[0], e[1]]).detach().cpu().numpy())
        neg.append(adj_orig[e[0], e[1]])
        
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score

@timeit
def get_gravity_roc_score_split(emb, adj_orig, edges_pos, edges_neg, device, split_size=100, pop_bias=5., normalization=True):
    emb = emb.detach().to(device)
    """
    dist = torch.cdist(x1, x2)
    x1 ( batch, dim_x, dim_y )
    y1 ( batch, dim_z, dim_y )
    dist = (batch, dim_x, dim_z)

    (100, 16), (10, 16) ->  (100, 10)
    
    """    
    emb_dim = emb.shape[1]
    mass = emb[:, emb_dim-1:emb_dim]
    
    if normalization:
        emb = F.normalize(emb[:,0:emb_dim-1], p=2.0, dim=1)
    else:             
        emb = emb[:,0:emb_dim-1]
        
    emb_splits = torch.split(emb, split_size, dim=0)    
    mass_splits = torch.split(mass, split_size, dim=0)
    adj_rec = None
    for idx, emb_split in enumerate(emb_splits):
        mass = mass_splits[idx].repeat(1, emb.shape[0])        
        
        tmp_adj_rec = torch.cdist(emb_split, emb)
        tmp_adj_rec = mass - torch.mul(pop_bias, tmp_adj_rec)
        if adj_rec is None:
            adj_rec = tmp_adj_rec
        else:
            adj_rec = torch.cat((adj_rec, tmp_adj_rec), dim=0)
        
    preds = [] 
    pos = []
    for e in edges_pos:
        preds.append(torch.sigmoid(adj_rec[e[0], e[1]]).detach().cpu().numpy())
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = [] 
    neg = [] 
    for e in edges_neg:
        preds_neg.append(torch.sigmoid(adj_rec[e[0], e[1]]).detach().cpu().numpy())
        neg.append(adj_orig[e[0], e[1]])
        
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score
