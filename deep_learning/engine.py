import monai
import torch
import numpy as np
from tqdm import tqdm
import gc
from monai.networks.utils import one_hot
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import seaborn as sns
from utils import evaluate, set_seed

class Engine:
  def __init__(self, network, optimizer, loss, train_loader, val_loader=None, scheduler=None, scheduler_step='epoch', device = None):
    self.device = device
    if self.device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.network = network.to(self.device)
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.scheduler_step = scheduler_step
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.loss = loss.to(self.device)

  def save(self, path, trainable=True, stats=None):
    save_dict = {
      'network': self.network.state_dict()
    }
    if stats is not None:
      save_dict['stats'] = stats
    if trainable:
      save_dict['optimizer'] = self.optimizer.state_dict()
      save_dict['scheduler'] = self.scheduler.state_dict()
    torch.save(save_dict, path)

  def load(self, path, return_stats = True):
    checkpoint = torch.load(path)
    self.network.load_state_dict(checkpoint['network'])
    if 'optimizer' in checkpoint:
      self.optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
      self.scheduler.load_state_dict(checkpoint['scheduler'])
    if return_stats:
      return checkpoint['stats']

  def train(self, epochs=1, save_path = 'best_cp.pt', results=None,  verbosity = True):
    if results is None:
      results = {
        "best_epoch":-1,
        "train": {"loss":[],"accuracy":[],"f1score":[],"conf_matrix":[]},
        "val": {"loss":[],"accuracy":[],"f1score":[],"conf_matrix":[]}
      }
      best_accuracy = 0
    else:
      best_accuracy = results['val']['accuracy'][results['best_epoch']]
    for epoch in range(results['best_epoch']+1, epochs):
      print(f"================== Epoch {epoch} / {epochs} ==================")
      epoch_loss, accuracy, f1 = 0, 0, 0
      all_labels = []
      all_preds = []
      self.network.train()
      for idx, batch in enumerate(tqdm(self.train_loader)):
        gc.collect()
        torch.cuda.empty_cache()
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        preds = self.network(images)
        preds = softmax(preds, dim=1)
        loss = self.loss(preds, one_hot(labels, 5))
        preds = torch.argmax(preds, 1).detach().cpu().numpy()
        labels = labels.cpu().numpy()
        all_labels.extend(labels)
        all_preds.extend(preds)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()
        if self.scheduler is not None and self.scheduler_step=='batch':
          self.scheduler.step()
      epoch_loss /= len(self.train_loader)
      accuracy, f1, cm = evaluate(all_labels, all_preds)
      val_loss, val_accuracy, val_f1, val_cm = self.test()
      print(f"\nTraining: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}, F1 Score = {sum(f1)/len(f1):.4f} : {f1}")
      print(f"Validation: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.4f}, F1 Score = {sum(val_f1)/len(val_f1):.4f} : {val_f1}")
      if verbosity:
        plt.figure(figsize=(7, 3))
        plt.subplot(1,2,1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Training Confusion Matrix')
        plt.subplot(1,2,2)
        sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Validation Confusion Matrix')
        plt.show()
      results['train']['loss'].append(epoch_loss)
      results['train']['accuracy'].append(accuracy)
      results['train']['f1score'].append(f1)
      results['train']['conf_matrix'].append(cm)
      results['val']['loss'].append(val_loss)
      results['val']['accuracy'].append(val_accuracy)
      results['val']['f1score'].append(val_f1)
      results['val']['conf_matrix'].append(val_cm)
      if self.scheduler is not None and self.scheduler_step=='epoch':
        self.scheduler.step()
      if val_accuracy > best_accuracy:
        results["best_epoch"] = epoch
        best_accuracy = val_accuracy
        self.save(save_path, stats=results)
      print(f"==============================================================")
    return results

  @torch.no_grad()
  def test(self, dataloader=None):
      if dataloader is None:
        dataloader = self.val_loader
      self.network.eval()
      loss, accuracy, f1 = 0, 0, 0
      all_labels = []
      all_preds = []
      for idx, batch in enumerate(dataloader):
          gc.collect()
          torch.cuda.empty_cache()
          images = batch['image'].to(self.device)
          labels = batch['label'].to(self.device)
          preds = self.network(images)
          preds = softmax(preds, dim=1)
          loss += self.loss(preds, one_hot(labels, 5)).item()
          preds = torch.argmax(preds, 1).cpu().numpy()
          labels = labels.cpu().numpy()
          all_labels.extend(labels)
          all_preds.extend(preds)
      loss /= len(dataloader)
      accuracy, f1, cm = evaluate(all_labels, all_preds)
      return loss, accuracy, f1, cm
