"""
Show resolved
This module defines the following routines used by the 'train' step:
- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Dict, Any
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, ProgressBar
from skorch.hf import HuggingfacePretrainedTokenizer
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
# for model hosting and requirements
from pathlib import Path
import transformers
import skorch
import sklearn
import torch

TOKENIZER = "distilbert-base-uncased"
PRETRAINED_MODEL = "distilbert-base-uncased"

# model hyper-parameters
OPTMIZER = torch.optim.AdamW
LR = 5e-5
MAX_EPOCHS = 3
CRITERION = nn.CrossEntropyLoss
BATCH_SIZE = 128

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class BertModule(nn.Module):
    def __init__(self, name, num_labels):
        super().__init__()
        self.name = name
        self.num_labels = num_labels

        self.reset_weights()

    def reset_weights(self):
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels
        )

    def forward(self, **kwargs):
        pred = self.bert(**kwargs)
        return pred.logits

def lr_schedule(current_step, num_training_steps=41940):
    factor = float(num_training_steps - current_step) / float(max(1, num_training_steps))
    assert factor > 0
    return factor

def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    pipeline = Pipeline([
        ('tokenizer', HuggingfacePretrainedTokenizer(TOKENIZER)),
        ('net', NeuralNetClassifier(
            BertModule,
            module__name=PRETRAINED_MODEL,
            module__num_labels=7, #len(set(y_train)),
            optimizer=OPTMIZER,
            lr=LR,
            max_epochs=MAX_EPOCHS,
            criterion=CRITERION,
            batch_size=BATCH_SIZE,
            iterator_train__shuffle=True,
            device=DEVICE,
            callbacks=[
                 LRScheduler(LambdaLR, lr_lambda=lr_schedule, step_every='batch'),
                 ProgressBar(),
            ],
        )),
    ])
    return pipeline
