import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PromptDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        _, preds = torch.max(outputs.logits, dim=1)
        loss = outputs.loss
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    accuracy = correct_predictions.double() / total_predictions
    return accuracy, np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            _, preds = torch.max(outputs.logits, dim=1)
            loss = outputs.loss
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            losses.append(loss.item())
    accuracy = correct_predictions.double() / total_predictions
    return accuracy, np.mean(losses)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

train_data_loader = create_data_loader(train_df, tokenizer, 128, 16)
val_data_loader = create_data_loader(val_df, tokenizer, 128, 16)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
EPOCHS = 50

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device)
    val_acc, val_loss = eval_model(model, val_data_loader, device)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}')

# Save the model
model.save_pretrained("Map_Generation_Model")
tokenizer.save_pretrained("Map_Generation_Model")
