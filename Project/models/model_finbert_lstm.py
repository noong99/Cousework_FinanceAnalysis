import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import  numpy as np
# from pytorch_pretrained_bert import BertTokenizr
# from bertModel import BertClassification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel

file_path = '../data/news_data_sampled.csv'
df = pd.read_csv(file_path)

texts = df['Title_Text'].values
labels = df['sentiment'].values
scores = df['sentimentScore'].values

xtrain, xtemp, ytrain, ytemp, scores_train, scores_temp= train_test_split(texts, labels, scores, test_size = 0.4, random_state = 129, stratify = labels)
xtest, xvalid, ytest, yvalid, scores_valid, scores_test= train_test_split(xtemp, ytemp, scores_temp, test_size=0.5, random_state = 129, stratify = ytemp)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

class FinDataset(Dataset):
    def __init__(self, texts, scores, labels):
        self.x = pd.Series(texts) 
        self.scores = pd.Series(scores) 
        self.y = pd.Series(labels) 

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        encoded = tokenizer(self.x.iloc[idx], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'score': torch.tensor(self.scores.iloc[idx], dtype=torch.float32),
            'label': torch.tensor(self.y.iloc[idx], dtype=torch.long)
        }

train_dataset = FinDataset(xtrain, scores_train, ytrain)
test_dataset = FinDataset(xtest, scores_test, ytest)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class FinBERTWithLSTM(nn.Module):
    def __init__(self):
        super(FinBERTWithLSTM, self).__init__()
        self.finbert = AutoModel.from_pretrained("ProsusAI/finbert", num_labels = 3)
        self.lstm = nn.LSTM(input_size=768, hidden_size = 128, batch_first=True)
        self.fc = nn.Linear(128 + 1, 3) 
        
    def forward(self, input_ids, attention_mask, scores):
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(outputs.last_hidden_state)
        combined = torch.cat((lstm_output[:, -1, :], scores.unsqueeze(1)), dim=1)
        logits = self.fc(combined)
        return logits
    
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        scores = batch['score']
        labels = batch['label']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, scores)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            scores = batch['score']
            labels = batch['label']

            outputs = model(input_ids, attention_mask, scores)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(test_loader), accuracy

model = FinBERTWithLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

epochs = 5 
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")