import torch
from sklearn.metrics import confusion_matrix

def calculate_confusion_aware_weights(targets, preds, num_classes, smoothing_factor=0.1):
    cm = confusion_matrix(targets, preds, labels=np.arange(num_classes))
    
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    
    class_errors = fp + fn
    
    total_errors = class_errors.sum()
    if total_errors == 0:
        return torch.ones(num_classes, dtype=torch.float)
        
    weights = class_errors / total_errors
    
    weights += smoothing_factor
    
    return torch.tensor(weights, dtype=torch.float)