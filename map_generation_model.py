import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

# Generate the CSV file with detailed geography and non-geography prompts
geo_prompts = [
    "Show me the borders around latitude {} and longitude {}",
    "Display the rivers and lakes at latitude {} and longitude {}",
    "Highlight urban areas and railroads at latitude {} and longitude {}",
    "Show the mountains and rivers near latitude {} and longitude {}",
    "Show the coastlines and major islands around latitude {} and longitude {}",
    "Display countries and state boundaries near latitude {} and longitude {}",
    "Highlight urban areas and major highways around latitude {} and longitude {}",
    "Show the geographical features including deserts and forests around latitude {} and longitude {}",
    "Plot the political and physical boundaries at latitude {} and longitude {}",
    "Show the elevation profile and topographical features at latitude {} and longitude {}",
    "Show me the administrative borders around latitude {} and longitude {}",
    "Display the coastline and major rivers at latitude {} and longitude {}",
    "Highlight urban infrastructure at latitude {} and longitude {}",
    "Show me natural features such as mountains and rivers near latitude {} and longitude {}",
    "Show the coastal lines and major islands around latitude {} and longitude {}",
    "Display country and state borders near latitude {} and longitude {}",
    "Highlight urban areas and transportation networks around latitude {} and longitude {}",
    "Show the natural landforms like deserts and forests around latitude {} and longitude {}",
    "Plot the boundaries and geographic features at latitude {} and longitude {}",
    "Show the elevation changes and terrain details at latitude {} and longitude {}",
    "Show the administrative boundaries at latitude {} and longitude {}",
    "Display the waterways at latitude {} and longitude {}",
    "Highlight cities and railroads at latitude {} and longitude {}",
    "Show the physical geography such as mountains and lakes near latitude {} and longitude {}",
    "Show the coastal regions and islands around latitude {} and longitude {}",
    "Display international borders and administrative divisions near latitude {} and longitude {}",
    "Highlight metropolitan areas and transportation routes around latitude {} and longitude {}",
    "Show environmental features like forests and mountains around latitude {} and longitude {}",
    "Plot detailed boundaries and features at latitude {} and longitude {}",
    "Show the topographical features and elevation data at latitude {} and longitude {}",
    "Show me state and country borders around latitude {} and longitude {}",
    "Display the coastline and river systems at latitude {} and longitude {}",
    "Highlight main urban centers and railways at latitude {} and longitude {}",
    "Show natural formations like hills and rivers near latitude {} and longitude {}",
    "Show the coastal outlines and islands at latitude {} and longitude {}",
    "Display borders and administrative regions near latitude {} and longitude {}",
    "Highlight key urban areas and transport links at latitude {} and longitude {}",
    "Show land features like deserts and mountains around latitude {} and longitude {}",
    "Plot geographic boundaries and natural features at latitude {} and longitude {}",
    "Show contour lines and elevation changes at latitude {} and longitude {}",
    "Show me national borders and regional divisions around latitude {} and longitude {}",
    "Display the waterways including lakes and rivers at latitude {} and longitude {}",
    "Highlight urban development and rail infrastructure at latitude {} and longitude {}",
    "Show the physical landscape such as mountains and valleys near latitude {} and longitude {}",
    "Show the coastal regions and islands at latitude {} and longitude {}",
    "Display country divisions and state boundaries near latitude {} and longitude {}",
    "Highlight significant urban zones and transport corridors around latitude {} and longitude {}",
    "Show environmental geography like forests and mountain ranges around latitude {} and longitude {}",
    "Plot precise boundaries and geographic elements at latitude {} and longitude {}",
    "Show detailed elevation and terrain data at latitude {} and longitude {}",
    "Show territorial borders and state lines around latitude {} and longitude {}",
    "Display the coastline and major rivers at latitude {} and longitude {}",
    "Highlight urban regions and rail lines at latitude {} and longitude {}",
    "Show natural geography such as mountain ranges and lakes near latitude {} and longitude {}",
    "Show the coastline features and islands at latitude {} and longitude {}",
    "Display political boundaries and administrative areas near latitude {} and longitude {}",
    "Highlight main urban infrastructures and transportation networks at latitude {} and longitude {}",
    "Show geographic features like deserts and forests around latitude {} and longitude {}",
    "Plot detailed political and natural boundaries at latitude {} and longitude {}",
    "Show contour maps and elevation profiles at latitude {} and longitude {}",
    "Show me international and local borders around latitude {} and longitude {}",
    "Display coastal regions and river basins at latitude {} and longitude {}",
    "Highlight urban areas and main railways at latitude {} and longitude {}",
    "Show physical features like mountains and valleys near latitude {} and longitude {}",
    "Show the coastlines and islands at latitude {} and longitude {}",
    "Display borders of countries and states near latitude {} and longitude {}",
    "Highlight urban development and transportation systems around latitude {} and longitude {}",
    "Show natural features including forests and mountains at latitude {} and longitude {}",
    "Plot precise geographic and political boundaries at latitude {} and longitude {}",
    "Show topographic elevation data at latitude {} and longitude {}",
    "Show borders and state divisions around latitude {} and longitude {}",
    "Display the coastline and main rivers at latitude {} and longitude {}",
    "Highlight significant urban centers and railway systems at latitude {} and longitude {}",
    "Show natural formations like mountains and rivers near latitude {} and longitude {}",
    "Show coastal lines and islands at latitude {} and longitude {}",
    "Display borders and administrative boundaries near latitude {} and longitude {}",
    "Highlight urban areas and transportation links at latitude {} and longitude {}",
    "Show landforms such as deserts and mountains at latitude {} and longitude {}",
    "Plot geographic boundaries and features at latitude {} and longitude {}",
    "Show detailed contour and elevation at latitude {} and longitude {}",
    "Show national and state borders at latitude {} and longitude {}",
    "Display the coastline and river systems at latitude {} and longitude {}",
    "Highlight major urban areas and rail infrastructure at latitude {} and longitude {}",
    "Show natural geography like hills and rivers near latitude {} and longitude {}",
    "Show the coastline features and islands at latitude {} and longitude {}",
    "Display international borders and state divisions near latitude {} and longitude {}",
    "Highlight key urban centers and transport routes at latitude {} and longitude {}",
    "Show environmental features like forests and mountains around latitude {} and longitude {}",
    "Plot detailed boundaries and natural features at latitude {} and longitude {}",
    "Show elevation and topographical features at latitude {} and longitude {}"
]

non_geo_prompts = [
    "How's the weather today?",
    "What is the capital of France?",
    "What's the population of New York?",
    "Give me a list of countries in Europe",
    "Tell me a joke",
    "Who won the game last night?",
    "What's the time now?",
    "Play some music",
    "Set an alarm for 7 AM",
    "What is the latest news?",
    "What is your favorite book?",
    "What is the meaning of life?",
    "How do I cook pasta?",
    "What's your favorite movie?",
    "Tell me a story",
    "What is 2 + 2?",
    "How do I get to the nearest supermarket?",
    "What is the weather forecast for tomorrow?",
    "How do I change a tire?",
    "What's the best way to learn Python?",
    "What's your favorite color?",
    "What's the best pizza topping?",
    "How do I make a sandwich?",
    "What's the best way to exercise?",
    "How do I meditate?",
    "What are the benefits of yoga?",
    "How do I improve my memory?",
    "What's the best book you've read?",
    "How do I learn to play guitar?",
    "What is the stock market doing today?",
    "What are the latest trends in fashion?",
    "How do I bake a cake?",
    "What's the best way to save money?",
    "How do I start a garden?",
    "What are the symptoms of a cold?",
    "How do I improve my sleep?",
    "What is the best diet?",
    "How do I quit smoking?",
    "What is the history of the internet?",
    "How do I start a blog?",
    "What is the best way to travel on a budget?",
    "How do I stay motivated?",
    "What are the best apps for productivity?",
    "How do I improve my public speaking skills?",
    "What is the best way to learn a new language?",
    "What are the best career options?",
    "How do I negotiate a salary?",
    "What are the best investment strategies?",
    "How do I build a personal brand?",
    "What are the best ways to network?",
    "How do I write a resume?",
    "What are the best interview tips?",
    "How do I stay organized?",
    "What are the best time management techniques?",
    "How do I set goals?",
    "What are the benefits of volunteering?",
    "How do I improve my writing skills?",
    "What is the best way to handle stress?",
    "How do I develop a growth mindset?",
    "What are the best ways to relax?",
    "How do I improve my relationships?",
    "What are the best communication skills?",
    "How do I practice mindfulness?",
    "What are the best leadership qualities?",
    "How do I build self-confidence?",
    "What are the benefits of positive thinking?",
    "How do I overcome procrastination?",
    "What are the best ways to stay healthy?",
    "How do I create a workout plan?",
    "What are the best foods for energy?",
    "How do I boost my immune system?",
    "What are the best ways to stay hydrated?",
    "How do I create a balanced diet?",
    "What are the best ways to lose weight?",
    "How do I maintain a healthy weight?",
    "What are the best exercises for beginners?",
    "How do I build muscle?",
    "What are the best cardio workouts?",
    "How do I improve my flexibility?",
    "What are the benefits of strength training?",
    "How do I reduce belly fat?",
    "What are the best ways to recover after a workout?",
    "How do I stay motivated to exercise?",
    "What are the best ways to improve endurance?",
    "How do I prevent injuries during exercise?",
    "What are the best warm-up exercises?",
    "How do I cool down after a workout?",
    "What are the best ways to track fitness progress?",
    "How do I create a fitness routine?",
    "What are the benefits of HIIT workouts?",
    "How do I incorporate rest days into my fitness plan?",
    "What are the best ways to stay active at home?",
    "How do I make exercise fun?",
    "What are the best fitness challenges?",
    "How do I set fitness goals?",
    "What are the best exercises for weight loss?",
    "How do I stay consistent with my workouts?",
    "What are the benefits of group fitness classes?",
    "How do I find a workout buddy?",
    "What are the best online fitness resources?"
]

data = []
for i in range(1000):
    if random.random() < 0.5:
        prompt = random.choice(geo_prompts).format(random.uniform(-90, 90), random.uniform(-180, 180))
        label = 1
    else:
        prompt = random.choice(non_geo_prompts)
        label = 0
    data.append([prompt, label])

df = pd.DataFrame(data, columns=['text', 'label'])
df.to_csv('geo_prompts.csv', index=False)
print("geo_prompts.csv file created with 1000 rows of data.")

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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load the dataset
from sklearn.model_selection import train_test_split
df = pd.read_csv('geo_prompts.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PromptDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

BATCH_SIZE = 16
MAX_LEN = 128

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fine-tune the model
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Training loop
for epoch in range(EPOCHS):
    model = model.train()
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate the model
    model = model.eval()
    val_loss = 0
    val_accuracy = 0
    for batch in val_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        loss = outputs.loss
        val_loss += loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        val_accuracy += accuracy

    val_loss /= len(val_data_loader)
    val_accuracy /= len(val_data_loader)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the model
model.save_pretrained("fine-tuned-geo-classifier")
tokenizer.save_pretrained("fine-tuned-geo-classifier")

# Load the fine-tuned model for later use
geo_classification_pipeline = pipeline("text-classification", model="fine-tuned-geo-classifier", tokenizer="fine-tuned-geo-classifier")

# Sample CSV data for map generation
csv_data = {
    "Latitude": [40.7128, 34.0522, 41.8781],
    "Longitude": [-74.0060, -118.2437, -87.6298],
    "City": ["New York", "Los Angeles", "Chicago"],
    "Country": ["USA", "USA", "USA"],
    "Population": [8419000, 3980000, 2716000],
    "Elevation": [10, 71, 181]
}
df = pd.DataFrame(csv_data)

def parse_prompt(prompt):
    parsed_data = {
        "features": [],
        "csv_files": [df],  # Use the sample data
        "longitude": None,
        "latitude": None
    }

    classification_result = geo_classification_pipeline(prompt)[0]
    if classification_result['label'] == 'LABEL_0':  # Assuming LABEL_0 is non-geography
        print("Don't forget that this is a cartography app.")
        return parsed_data

    # Improved extraction of latitude and longitude
    lat_long_matches = re.findall(r"latitude\s(-?\d+(\.\d+)?)\sand\slongitude\s(-?\d+(\.\d+)?)", prompt, re.IGNORECASE)
    if lat_long_matches:
        parsed_data["latitude"] = float(lat_long_matches[0][0])
        parsed_data["longitude"] = float(lat_long_matches[0][2])

    if not parsed_data["longitude"] or not parsed_data["latitude"]:
        print("Please provide valid longitude and latitude in your prompt.")
        return parsed_data

    if "borders" in prompt.lower():
        parsed_data["features"].append("BORDERS")
    if "coastline" in prompt.lower():
        parsed_data["features"].append("COASTLINE")
    if "lakes" in prompt.lower():
        parsed_data["features"].append("LAKES")
    if "rivers" in prompt.lower():
        parsed_data["features"].append("RIVERS")
    if "country" in prompt.lower():
        parsed_data["features"].append("NaturalEarthFeature('cultural', 'admin_0_countries', '10m')")
    if "urban" in prompt.lower():
        parsed_data["features"].append("NaturalEarthFeature('cultural', 'urban_areas', '10m')")
    if "railroads" in prompt.lower():
        parsed_data["features"].append("NaturalEarthFeature('cultural', 'railroads', '10m')")

    return parsed_data

def generate_map(prompt):
    parsed_data = parse_prompt(prompt)

    if not parsed_data["longitude"] or not parsed_data["latitude"]:
        print("Please provide valid longitude and latitude in your prompt.")
        return

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([parsed_data["longitude"] - 10, parsed_data["longitude"] + 10, parsed_data["latitude"] - 10, parsed_data["latitude"] + 10], crs=ccrs.PlateCarree())

    for feature_name in parsed_data["features"]:
        try:
            if "NaturalEarthFeature" in feature_name:
                feature = eval(f"cfeature.{feature_name}")
                ax.add_feature(feature, edgecolor='black')
            else:
                feature = getattr(cfeature, feature_name)
                ax.add_feature(feature)
        except Exception as e:
            print(f"Error adding feature {feature_name}: {e}")

    for df in parsed_data["csv_files"]:
        sc = ax.scatter(df['Longitude'], df['Latitude'], c=df['Population'], cmap='viridis', s=50, edgecolor='k', transform=ccrs.PlateCarree(), label='Population')
        cb = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05)
        cb.set_label('Population')

        for i, row in df.iterrows():
            ax.text(row['Longitude'], row['Latitude'], row['City'], fontsize=8, transform=ccrs.PlateCarree())

    ax.gridlines(draw_labels=True)
    plt.show()

# Main loop for terminal input
while True:
    prompt = input("Enter your prompt (or 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break
    generate_map(prompt)
