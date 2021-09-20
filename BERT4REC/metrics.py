import torch 

def masked_accuracy(y_pred, y_label, mask):    
    _, predicted = torch.max(y_pred, 1)

    predicted = torch.masked_select(predicted, mask)
    y_label = torch.masked_select(y_label, mask)
    
    accuracy = (y_label == predicted).double().mean()

    return accuracy

def masked_recall_at_k(y_pred, y_label, mask, k):
    y_label_at_k = y_label.unsqueeze(1).repeat(1,k)
    _, predicted_at_k = torch.topk(y_pred, k) 

    recall_at_k = (predicted_at_k == y_label_at_k).view(-1, k)
    recall_at_k, _ = torch.max(recall_at_k,1)

    recall_at_k = torch.masked_select(recall_at_k, mask)
    recall_at_k = recall_at_k.double().mean()

    return recall_at_k
