import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import  numpy as np
# from pytorch_pretrained_bert import BertTokenizr
# from bertModel import BertClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

file_path = '../data/news_data_sampled.csv'
df = pd.read_csv(file_path)

texts = df['Title_Text'].values
labels = df['sentiment'].values
scores = df['sentimentScore'].values

xtrain, xtemp, ytrain, ytemp, scores_train, scores_temp= train_test_split(texts, labels, scores, test_size = 0.4, random_state = 129, stratify = labels)
xtest, xvalid, ytest, yvalid, scores_valid, scores_test= train_test_split(xtemp, ytemp, scores_temp, test_size=0.5, random_state = 129, stratify = ytemp)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, scores, tokenizer, max_len=128):
        self.x = texts 
        self.y = labels 
        self.scores = scores 
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.x) 

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        score = self.scores[index]
        encoding = self.tokenizer(
            x,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float32),
        }

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_dataset = SentimentDataset(xtrain, ytrain, scores_train, tokenizer)
valid_dataset = SentimentDataset(xvalid, yvalid, scores_valid, tokenizer)
test_dataset = SentimentDataset(xtest, ytest, scores_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 1, output_dim)  

    def forward(self, input_ids, scores):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        hidden_with_score = torch.cat((hidden.squeeze(0), scores.unsqueeze(1)), dim=1)
        logits = self.fc(hidden_with_score)
        return logits

vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 128
output_dim = len(set(labels))

model = LSTMModel(embedding_dim, hidden_dim, output_dim, vocab_size)

def weighted_loss(outputs, labels, scores):
    weights = 1 + scores
    loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
    weighted_loss = (loss * weights).mean() 
    return weighted_loss

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, data_loader, optimizer):
    model.train()
    losses = []
    for batch in data_loader:
        input_ids = batch["input_ids"]
        labels = batch["label"]
        scores = batch["score"]

        outputs = model(input_ids, scores)
        loss = weighted_loss(outputs, labels, scores)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(losses)

def evaluate_model(model, data_loader):
    model.eval()
    losses = []
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            labels = batch["label"]
            scores = batch["score"]

            outputs = model(input_ids, scores)
            loss = weighted_loss(outputs, labels, scores)
            losses.append(loss.item())

            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return np.mean(losses), accuracy_score(true_labels, predictions)

for epoch in range(5):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, val_acc = evaluate_model(model, valid_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

test_loss, test_acc = evaluate_model(model, test_loader)