import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import multiprocessing
import numpy as np
import pandas as pd

file_path = 'news_data_sampled.csv'
df = pd.read_csv(file_path)

# Encode the sentiment column
# 0: Negative, 1: Neutral, 2: Positive
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])
df['sentiment'].value_counts() #sentiment is means label

# Set Title_Text as texts, sentiment as labels
texts = df['Title_Text'].values
labels = df['sentiment'].values
scores = df['sentimentScore'].values

# Split the data into train, validation, test set
# Set train:valid:test = 6:2:2 and apply stratify
xtrain, xtemp, ytrain, ytemp, scores_train, scores_temp= train_test_split(texts, labels, scores, test_size = 0.4, random_state = 129, stratify = labels)
xtest, xvalid, ytest, yvalid, scores_valid, scores_test= train_test_split(xtemp, ytemp, scores_temp, test_size=0.5, random_state = 129, stratify = ytemp)

# Check how many data in one each dataset
print(f"Train size: {len(xtrain)}")
print(f"Validation size: {len(xvalid)}")
print(f"Train size: {len(xtest)}")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def tokenize_function(x):
    return tokenizer(list(x), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_function(xtrain)
val_encodings = tokenize_function(xvalid)
test_encodings = tokenize_function(xtest)

# Define Pytorch Dataset
class FinDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.scores = scores

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item['scores'] = torch.tensor(self.scores[idx], dtype = torch.float)
        return item
    
# Apply SMOTE
smote = SMOTE(random_state=129)
train_features = train_encodings['input_ids'].numpy()
ytrain_resampled, y_resampled = smote.fit_resample(train_features, ytrain)

# Expand Attention mask and Token Type IDs
attention_mask_resampled = np.tile(
    train_encodings["attention_mask"].numpy(),
    (len(ytrain_resampled) // len(train_encodings["attention_mask"]) + 1, 1)
)[:len(ytrain_resampled)]

token_type_ids_resampled = np.tile(
    train_encodings["token_type_ids"].numpy(),
    (len(ytrain_resampled) // len(train_encodings["token_type_ids"]) + 1, 1))[:len(ytrain_resampled)]

# Adjust data size
num_samples = 4000

train_encodings_resampled = {
    "input_ids": torch.tensor(ytrain_resampled[:num_samples]),
    "attention_mask": torch.tensor(attention_mask_resampled[:num_samples]),
    "token_type_ids": torch.tensor(token_type_ids_resampled[:num_samples]),
}

y_resampled = y_resampled[:num_samples]

train_dataset = FinDataset(train_encodings_resampled, y_resampled)
val_dataset = FinDataset(val_encodings, yvalid)
test_dataset = FinDataset(test_encodings, ytest)

# Check these shapes are same or not
print(f"Input IDs shape: {train_encodings_resampled['input_ids'].shape}")
print(f"Attention Mask shape: {train_encodings_resampled['attention_mask'].shape}")
print(f"Token Type IDs shape: {train_encodings_resampled['token_type_ids'].shape}")
print(f"Labels shape: {len(y_resampled)}")
print(f"Train Dataset size: {len(train_dataset)}")

# Compute class weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define custom loss function
# 
def compute_loss(outputs, labels, scores):
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    weighted_loss = loss_func(outputs.logits, labels)
    # apply weight using 'sentimentScore'
    adjusted_loss = (scores * weighted_loss).mean()
    return adjusted_loss

# Define model
# Model Configuration
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels = 3)

# Set the training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)

# Define evaluation metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Define Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        scores = inputs.pop("scores")  # sentimentScore
        outputs = model(**inputs)
        loss = compute_loss(outputs, labels, scores)  # Custom loss function
        return (loss, outputs) if return_outputs else loss

# Tracking
from transformers import TrainerCallback

class CustomProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Epoch {state.epoch}: {logs}")

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics=compute_metrics
)

# Training model
trainer.train()

# Model Evaluation
val_results = trainer.evaluate(eval_dataset = val_dataset)
print("Validation Results: ", val_results)

test_predictions = trainer.predict(test_dataset)
ytest_pred = np.argmax(test_predictions.predictions, axis = 1)
print(classification_report(ytest, ytest_pred, target_names = ["Negative", "Neutral", "Positive"]))