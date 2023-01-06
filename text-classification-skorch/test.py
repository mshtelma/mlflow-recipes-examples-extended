# %% Importing the dependencies we need
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

# %%
# Choose a tokenizer and BERT model that work together
TOKENIZER = "distilbert-base-uncased"
PRETRAINED_MODEL = "distilbert-base-uncased"

# model hyper-parameters
OPTMIZER = torch.optim.AdamW
LR = 5e-5
MAX_EPOCHS = 3
CRITERION = nn.CrossEntropyLoss
BATCH_SIZE = 8

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% Load the dataset, define features & labels and split
dataset = fetch_20newsgroups()

print(dataset.DESCR.split('Usage')[0])

dataset.target_names

X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test, = train_test_split(X, y, stratify=y, random_state=0)
num_training_steps = MAX_EPOCHS * (len(X_train) // BATCH_SIZE + 1)

# %%
# Defining learning rate scheduler & BERT in nn.Module

def lr_schedule(current_step):
    factor = float(num_training_steps - current_step) / float(max(1, num_training_steps))
    assert factor > 0
    return factor

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

# %% Chaining tokenizer and BERT in one pipeline
pipeline = Pipeline([
    ('tokenizer', HuggingfacePretrainedTokenizer(TOKENIZER)),
    ('net', NeuralNetClassifier(
        BertModule,
        module__name=PRETRAINED_MODEL,
        module__num_labels=len(set(y_train)),
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

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# %% Training
pipeline.fit(X_train, y_train)

# %% Evaluate the model
with torch.inference_mode():
    y_pred = pipeline.predict(X_test)

accuracy_score(y_test, y_pred)
