from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch, random, numpy as np, monai


def evaluate(labels, preds):
  accuracy = accuracy_score(labels, preds)
  f1 = f1_score(labels, preds, average=None, labels=[0,1,2,3,4], zero_division=0)    
  cm = confusion_matrix(labels, preds)  
  return accuracy, f1, cm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)
