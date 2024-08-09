import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import torch
import random
from mrmr import mrmr_classif
from conventional.feature_extraction import train_features, val_features, test_features, train_classes, val_classes, test_classes
from utils import evaluate, set_seed


def apply_mrmr(train_features, train_classes, val_features, test_features, k=1000):
  feature_columns = [f'feature_{i}' for i in range(train_features.shape[1])]
  train_features = pd.DataFrame(train_features, columns=feature_columns)
  train_classes = pd.Series(train_classes, name='target')
  selected_features = mrmr_classif(X=train_features, y=train_classes, K=k, show_progress=True)
  train_features = train_features[selected_features].to_numpy()
  val_features = pd.DataFrame(val_features, columns=feature_columns)
  val_features = val_features[selected_features].to_numpy()
  test_features = pd.DataFrame(test_features, columns=feature_columns)
  test_features = test_features[selected_features].to_numpy()
  return train_features, val_features, test_features

set_seed(42)

smote=SMOTE()
train_features_balanced, train_classes_balanced = smote.fit_resample(train_features, train_classes)

scaler = StandardScaler()
train_features_balanced = scaler.fit_transform(train_features_balanced)
train_features = scaler.transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

n_components = 1250
pca = PCA(n_components=n_components)
train_features_balanced = pca.fit_transform(train_features_balanced)
train_features = pca.transform(train_features)
val_features = pca.transform(val_features)
test_features = pca.transform(test_features)

k = 250
kbest = SelectKBest(score_func=f_classif, k=k)
train_features_balanced = kbest.fit_transform(train_features_balanced, train_classes_balanced)
train_features = kbest.transform(train_features)
val_features = kbest.transform(val_features)
test_features = kbest.transform(test_features)

# train_features_balanced, val_features, test_features = apply_mrmr(train_features_balanced, train_classes, val_features, test_features, k)

model = SVC(kernel='rbf', C=1.2)
model.fit(train_features_balanced, train_classes_balanced)
