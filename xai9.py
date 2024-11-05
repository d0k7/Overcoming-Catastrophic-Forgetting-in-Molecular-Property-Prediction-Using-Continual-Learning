import os
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# Define model and dataset paths
model_paths = {
    "C2C": "/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/C2C_BERT_Model"
}

dataset_paths = {
    "C2C": "/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/BBBPC2C.csv"
}

local_tokenizer_path = "/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/BERT_Base_Uncased_Tokenizer"

# Initialize tokenizer
def download_tokenizer_with_retries(pretrained_model_name, retries=5, delay=5):
    for attempt in range(retries):
        try:
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, force_download=True)
            return tokenizer
        except Exception as e:
            print(f"Error downloading the tokenizer: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise ValueError("Tokenizer could not be initialized after several attempts. Exiting.")

try:
    if os.path.exists(local_tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
    else:
        tokenizer = download_tokenizer_with_retries('bert-base-uncased')
except ValueError as e:
    print(e)
    raise ValueError(f"Failed to load the tokenizer from local path: {e}")

# Load and tokenize dataset
def load_and_tokenize_dataset(path, tokenizer):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    df = pd.read_csv(path)
    if 'Canonical_SMILES' not in df.columns or 'Label' not in df.columns:
        raise KeyError("Dataset must contain 'Canonical_SMILES' and 'Label' columns")

    inputs = df['Canonical_SMILES'].tolist()
    labels = df['Label'].tolist()
    encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_mask, labels, inputs

# Dataset class for PyTorch DataLoader
class SMILESDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Load datasets
datasets = {key: load_and_tokenize_dataset(path, tokenizer) for key, path in dataset_paths.items() if key == "C2C"}

# Split data into training and validation sets
input_ids, attention_mask, labels, raw_inputs = datasets["C2C"]
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1)
train_mask, val_mask = train_test_split(attention_mask, test_size=0.1)

train_dataset = SMILESDataset(train_inputs, train_mask, train_labels)
val_dataset = SMILESDataset(val_inputs, val_mask, val_labels)

# Initialize model
model = BertForSequenceClassification.from_pretrained(model_paths["C2C"], num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch"
)

# Function to calculate cosine distance between attributions and baseline
def calculate_cosine_distance(attributions):
    baseline = np.zeros_like(attributions)
    distances = cosine_distances(attributions, baseline)
    return distances

# Function to aggregate data for visualization
def prepare_data_for_visualization(attributions, tokenizer, inputs):
    decoded_inputs = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs]
    attention_values = [attr.tolist() for attr in attributions]
    
    data = []
    for tokens, attention in zip(decoded_inputs, attention_values):
        for token, attention_score in zip(tokens, attention):
            data.append([token, attention_score])

    return data

# Function to create box plot for cosine distances
def visualize_results(results):
    plt.figure(figsize=(10, 6))
    plt.boxplot(results)
    plt.title('Cosine Distances of Attributions')
    plt.xlabel('XAI Methods')
    plt.ylabel('Cosine Distance')
    plt.show()

# LIME explanation function
def lime_explanation(model, tokenizer, inputs):
    model.eval()
    explainer = LimeTextExplainer(class_names=['Non-Toxic', 'Toxic'])

    def predict_fn(texts):
        encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=50)
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

    attributions = []
    for input_text in inputs:
        exp = explainer.explain_instance(input_text, predict_fn, num_features=10)
        tokenized_input = tokenizer.encode(input_text, max_length=50, truncation=True)
        temp_attr = np.zeros(50)  # assuming max_length=50
        for word, weight in exp.as_list():
            for idx, token in enumerate(tokenizer.convert_ids_to_tokens(tokenized_input)):
                if word in token:
                    if idx < 50:
                        temp_attr[idx] += weight
        attributions.append(temp_attr)
    
    return attributions

try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train and evaluate
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/LIME_fine_tuned_bert_model")

    # Load fine-tuned model
    model = BertForSequenceClassification.from_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/LIME_fine_tuned_bert_model")

    # Compute LIME explanations
    def compute_lime_explanations(model, tokenizer, raw_inputs):
        lime_attributions = lime_explanation(model, tokenizer, raw_inputs)
        return lime_attributions

    # Save LIME explanations to CSV with tokens
    def save_lime_explanations_to_csv(lime_attributions, inputs, path):
        decoded_inputs = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs]
        data = []
        for tokens, lime_scores in zip(decoded_inputs, lime_attributions):
            for token, lime_score in zip(tokens, lime_scores):
                data.append([token, lime_score])

        df = pd.DataFrame(data, columns=['Token', 'LIME Score'])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"LIME explanations saved to {path} successfully.")
        except Exception as e:
            print(f"Error saving LIME explanations: {e}")

    # Loop over models and datasets for LIME explanations computation
    results = {}
    computation_successful = True

    for key in ["C2C"]:
        inputs, attention_mask, labels, raw_inputs = datasets[key]

        lime_attributions = compute_lime_explanations(model, tokenizer, raw_inputs)

        csv_path = f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_XAI_Computed/{key}_LIME_Explanations.csv"
        save_lime_explanations_to_csv(lime_attributions, inputs, csv_path)

        results[key] = lime_attributions

    # Calculate cosine distances
    distances = {key: calculate_cosine_distance(np.array(result)) for key, result in results.items()}

    # Prepare data for visualization
    data = {key: prepare_data_for_visualization(np.array(results[key]), tokenizer, datasets[key][0]) for key in results}

    # Visualize results
    visualize_results(list(distances.values()))

except Exception as e:
    computation_successful = False
    print(f"An error occurred during training or LIME explanation computation: {e}")

if computation_successful:
    print("Computation and saving of LIME explanations completed successfully.")
else:
    print("Computation or saving of LIME explanations failed.")
