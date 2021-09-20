from torch.nn import functional as F

def masked_cross_entropy(y_pred, y_label, mask):
    loss = F.cross_entropy(y_pred, y_label)
    loss = loss * mask 
    return loss.sum() / (mask.sum())