import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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


def load_data_from_file(file_name, label):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    return pd.DataFrame({'text': lines, 'label': [label] * len(lines)})


def load_dataset():
    multitask_data = load_data_from_file('multitask_prompts.txt', 0)
    geospatial_data = load_data_from_file('geospatial_prompts.txt', 1)
    weather_data = load_data_from_file('weather_prompts.txt', 2)
    combined_df = pd.concat([multitask_data, geospatial_data, weather_data])
    return combined_df


df = load_dataset()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PromptDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

train_data_loader = create_data_loader(train_df, tokenizer, 128, 16)
val_data_loader = create_data_loader(val_df, tokenizer, 128, 16)


optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
EPOCHS = 50

for epoch in range(EPOCHS):
    total_train_loss = 0
    total_train_accuracy = 0
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_accuracy += accuracy.item()

    avg_train_loss = total_train_loss / len(train_data_loader)
    avg_train_accuracy = total_train_accuracy / len(train_data_loader)

    total_val_loss = 0
    total_val_accuracy = 0
    model.eval()
    with torch.no_grad():
        for batch in val_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

            total_val_loss += loss.item()
            total_val_accuracy += accuracy.item()

    avg_val_loss = total_val_loss / len(val_data_loader)
    avg_val_accuracy = total_val_accuracy / len(val_data_loader)

    print(f'Epoch {epoch + 1}/{EPOCHS}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_accuracy:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_accuracy:.4f}')


model.save_pretrained("Map_Generation_Model")
tokenizer.save_pretrained("Map_Generation_Model")
