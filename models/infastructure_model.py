import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import re
from geopy.geocoders import Nominatim

# Prompts for different tasks
prompts_dict = {
    "airports": [
        "Show airports in Turkey",
        "Display airports in France",
        "Highlight airports in Canada",
        "Show airports in Brazil",
        "Display airports in South Africa",
        "Show airports in Pakistan.",
        "Display airports in Vietnam.",
        "Highlight airports in Greece.",
        "Show airports in Switzerland.",
        "Display airports in Portugal.",
        "Show airports in Chile.",
        "Display airports in New Zealand.",
        "Highlight airports in Venezuela.",
        "Show airports in Algeria.",
        "Display airports in Philippines.",
        "Show airports in Finland.",
        "Display airports in Hungary.",
        "Highlight airports in Morocco.",
        "Show airports in Sweden.",
        "Display airports in Austria.",
        "Show airports in the Netherlands.",
        "Display airports in Belgium.",
        "Highlight airports in Czech Republic.",
        "Show airports in South Sudan.",
        "Display airports in Ivory Coast.",
        "Show airports in Nigeria.",
        "Display airports in Uzbekistan.",
        "Highlight airports in Kazakhstan.",
        "Show airports in Uganda.",
        "Display airports in Afghanistan.",

    ],
    "roads": [
        "Show roads in Europe",
        "Display roads in Africa",
        "Highlight roads in South America",
        "Show roads in Australia",
        "Display roads in Asia",
        "Show roads in Morocco.",
        "Display roads in Kenya.",
        "Highlight roads in Ireland.",
        "Show roads in Pakistan.",
        "Display roads in South Korea.",
        "Show roads in Ecuador.",
        "Display roads in Peru.",
        "Highlight roads in Taiwan.",
        "Show roads in Venezuela.",
        "Display roads in Algeria.",
        "Show roads in Norway.",
        "Display roads in Finland.",
        "Highlight roads in Thailand.",
        "Show roads in Saudi Arabia.",
        "Display roads in Romania.",
        "Show roads in Greece.",
        "Display roads in Poland.",
        "Highlight roads in Malaysia.",
        "Show roads in Austria.",
        "Display roads in Bulgaria.",
        "Show roads in Portugal.",
        "Display roads in Switzerland.",
        "Highlight roads in Ukraine.",
        "Show roads in Uzbekistan.",
        "Display roads in New Zealand.",

    ],
    "ports": [
        "Show ports in Turkey",
        "Display ports in USA",
        "Highlight ports in India",
        "Show ports in Brazil",
        "Display ports in China",
        "Show ports in Finland.",
        "Display ports in Vietnam.",
        "Highlight ports in Greece.",
        "Show ports in Switzerland.",
        "Display ports in Portugal.",
        "Show ports in Chile.",
        "Display ports in New Zealand.",
        "Highlight ports in Venezuela.",
        "Show ports in Algeria.",
        "Display ports in Philippines.",
        "Show ports in the Netherlands.",
        "Display ports in Belgium.",
        "Highlight ports in Norway.",
        "Show ports in Sweden.",
        "Display ports in Austria.",
        "Show ports in the Czech Republic.",
        "Display ports in Morocco.",
        "Highlight ports in Ivory Coast.",
        "Show ports in South Sudan.",
        "Display ports in Uganda.",
        "Show ports in Tanzania.",
        "Display ports in Kazakhstan.",
        "Highlight ports in Uzbekistan.",
        "Show ports in Afghanistan.",
        "Display ports in Zambia.",

    ],
    "power_plants": [
        "Show power plants in Germany",
        "Display power plants in Russia",
        "Highlight power plants in Canada",
        "Show power plants in Argentina",
        "Display power plants in South Korea",
        "Show power plants in the Netherlands.",
        "Display power plants in Belgium.",
        "Highlight power plants in Switzerland.",
        "Show power plants in Greece.",
        "Display power plants in Portugal.",
        "Show power plants in South Korea.",
        "Display power plants in Malaysia.",
        "Highlight power plants in Thailand.",
        "Show power plants in Kenya.",
        "Display power plants in Uganda.",
        "Show power plants in Tanzania.",
        "Display power plants in Nigeria.",
        "Highlight power plants in Morocco.",
        "Show power plants in Algeria.",
        "Display power plants in Uzbekistan.",
        "Show power plants in Kazakhstan.",
        "Display power plants in Afghanistan.",
        "Highlight power plants in Finland.",
        "Show power plants in Sweden.",
        "Display power plants in Norway.",
        "Show power plants in Venezuela.",
        "Display power plants in New Zealand.",
        "Highlight power plants in Chile.",
        "Show power plants in Ecuador.",
        "Display power plants in Peru.",

    ],
    "cities": [
        "Show cities in Japan",
        "Display cities in France",
        "Highlight cities in Egypt",
        "Show cities in Italy",
        "Display cities in Spain",
        "Show power plants in the Netherlands.",
        "Display power plants in Belgium.",
        "Highlight power plants in Switzerland.",
        "Show power plants in Greece.",
        "Display power plants in Portugal.",
        "Show power plants in South Korea.",
        "Display power plants in Malaysia.",
        "Highlight power plants in Thailand.",
        "Show power plants in Kenya.",
        "Display power plants in Uganda.",
        "Show power plants in Tanzania.",
        "Display power plants in Nigeria.",
        "Highlight power plants in Morocco.",
        "Show power plants in Algeria.",
        "Display power plants in Uzbekistan.",
        "Show power plants in Kazakhstan.",
        "Display power plants in Afghanistan.",
        "Highlight power plants in Finland.",
        "Show power plants in Sweden.",
        "Display power plants in Norway.",
        "Show power plants in Venezuela.",
        "Display power plants in New Zealand.",
        "Highlight power plants in Chile.",
        "Show power plants in Ecuador.",
        "Display power plants in Peru.",
    ],
    "non_relevant": [
        "What is the capital of Spain?",
        "Tell me a joke",
        "What's the time now?",
        "What is the best way to travel on a budget?",
        "How do I stay motivated?",
    ]
}

# Generate dataset
data = []
for task, prompts in prompts_dict.items():
    label = list(prompts_dict.keys()).index(task)  # Assign a unique label for each task
    for prompt in prompts:
        data.append([prompt, label])

# Duplicate and shuffle dataset
data *= 300
random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("multitask_prompts.csv", index=False)
print("multitask_prompts.csv file created.")

# Load and preprocess dataset
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Split dataset
from sklearn.model_selection import train_test_split

df = pd.read_csv("multitask_prompts.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(prompts_dict))

# Create DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PromptDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

BATCH_SIZE = 16
MAX_LEN = 128

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(EPOCHS):
    model = model.train()
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model = model.eval()
    val_accuracy = 0
    for batch in val_data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        val_accuracy += torch.sum(preds == labels).item() / len(labels)

    print(f"Epoch {epoch + 1}/{EPOCHS} Validation Accuracy: {val_accuracy / len(val_data_loader)}")

model.save_pretrained("fine-tuned-multitask-classifier")
tokenizer.save_pretrained("fine-tuned-multitask-classifier")

# Load the fine-tuned model
classification_pipeline = pipeline(
    "text-classification", model="fine-tuned-multitask-classifier", tokenizer="fine-tuned-multitask-classifier"
)

# Visualization Functions
def plot_geo_data(gdf, x_min, x_max, y_min, y_max, color="blue"):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    gdf.plot(ax=ax, color=color, transform=ccrs.PlateCarree())
    plt.show()

# Dataset Paths
datasets = {
    "airports": "geospatial-environmental-and-socioeconomic-data/3_Airports_ports/ne_10m_airports/ne_10m_airports.shp",
    "roads": "geospatial-environmental-and-socioeconomic-data/2_Roads_railroads/ne_10m_roads/ne_10m_roads.shp",
    "ports": "geospatial-environmental-and-socioeconomic-data/3_Airports_ports/ne_10m_ports/ne_10m_ports.shp",
    "power_plants": "geospatial-environmental-and-socioeconomic-data/4_globalpowerplantdatabasev120/global_power_plant_database.csv",
    "cities": "geospatial-environmental-and-socioeconomic-data/1_CITIES_landscan/ne_10m_populated_places/ne_10m_populated_places.shp",
}

# Main loop
def main():
    while True:
        prompt = input("Enter your prompt (or 'done' to exit): ")
        if prompt.lower() == "done":
            break

        result = classification_pipeline(prompt)[0]
        label = list(prompts_dict.keys())[int(result["label"])]
        print(f"Detected Task: {label}")

        if label in datasets:
            if label in ["airports", "roads", "ports", "cities"]:
                gdf = gpd.read_file(datasets[label])
                plot_geo_data(gdf, -40, 40, -10, 50, color="green")
            elif label == "power_plants":
                df = pd.read_csv(datasets[label])
                print("Visualize power plant data separately.")
            else:
                print("Task not visualized.")

if __name__ == "__main__":
    main()
