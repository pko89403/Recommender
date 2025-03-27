import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = (-0.5 / n_nodes) * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, ckpt_path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose 
        self.counter = 0
        self.best_score = None 
        self.early_stop = False 
        self.val_loss_min = np.Inf 
        self.delta = delta 
        self.ckpt_path = ckpt_path
        self.trace_func = trace_func
        
    def __call__(self, score, model):
        score = score
        
        if self.best_score is None:
            self.best_score = 0.0 
            self.save_checkpoint(score, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1 
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            model.load_state_dict(self.recover_best_model())
        else:
            self.best_score = score
            self.save_checkpoint(score, model) 
            self.counter = 0 
        return model

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.trace_func(f"Validation score updates ({self.best_score:.6f} --> {score:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.ckpt_path)
        self.best_score = score
        
    def recover_best_model(self):
        if self.verbose:
            self.trace_func(f"recover best score model ({self.best_score:.6f}). Loading model ...")
        return torch.load(self.ckpt_path)
        