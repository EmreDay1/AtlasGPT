import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def load_model(model_path, tokenizer_path):

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def classify_text(model, tokenizer, text):

    classification_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = classification_pipeline(text)
    return results

def main():
    model_path = 'Map_Generation_Model'
    tokenizer_path = 'Map_Generation_Model'

    # Load model and tokenizer
    model, tokenizer = load_model(model_path, tokenizer_path)

    print("Model loaded. Enter text to classify or 'exit' to quit.")
    while True:
        input_text = input("Enter your query: ")
        if input_text.lower() == 'exit':
            print("Exiting the program.")
            break

        # Classify the text
        result = classify_text(model, tokenizer, input_text)
        print(f"Classification result: {result}")

if __name__ == "__main__":
    main()
