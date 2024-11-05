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
    model.save_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/rollout_fine_tuned_bert_model")

    # Load fine-tuned model
    model = BertForSequenceClassification.from_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/rollout_fine_tuned_bert_model")

    # Custom Rollout implementation
    def compute_rollout_values(model, input_ids, attention_mask):
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attentions = outputs.attentions  # Tuple of attention matrices from all layers
            
            rollout_values = np.zeros(attentions[0].size()[-1])  # Initialize rollout values
            for i in range(len(attentions)):
                attention = attentions[i].mean(dim=1).squeeze(0)  # Mean attention weights across all heads
                rollout_values += attention.cpu().numpy().mean(axis=0)
            
            rollout_values /= len(attentions)  # Normalize by the number of layers
            
        return rollout_values

    # Save Rollout values to CSV
    def save_rollout_values_to_csv(rollout_values, raw_inputs, path):
        data = []
        for i in range(len(rollout_values)):
            input_text = raw_inputs[i]
            rollout_value = rollout_values[i]
            data.append([input_text] + rollout_value.tolist())
        
        df = pd.DataFrame(data, columns=['Input Text'] + [f'Rollout Value {i}' for i in range(len(rollout_values[0]))])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Rollout values saved to {path} successfully.")
        except Exception as e:
            print(f"Error saving Rollout values: {e}")

    # Loop over models and datasets for Rollout computation
    results = {}
    computation_successful = True

    for key in ["C2C"]:
        input_ids, attention_mask, labels, raw_inputs = datasets[key]
        rollout_values = []

        for i in range(len(input_ids)):
            input_id = input_ids[i].unsqueeze(0)
            attn_mask = attention_mask[i].unsqueeze(0)
            rollout_value = compute_rollout_values(model, input_id, attn_mask)
            rollout_values.append(rollout_value)

        csv_path = f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_XAI_Computed/{key}_Rollout_Values.csv"
        save_rollout_values_to_csv(rollout_values, raw_inputs, csv_path)

        results[key] = rollout_values

    # Step 1: Calculating Metrics (Cosine Distance)
    def calculate_cosine_distance(attributions):
        attributions_flat = attributions.reshape(attributions.shape[0], -1)
        baseline = np.zeros_like(attributions_flat)
        distances = cosine_distances(attributions_flat, baseline)
        return distances

    # Calculate cosine distances for the rollout values
    distances = calculate_cosine_distance(np.array(rollout_values))
    print("Cosine distances calculated successfully.")

    # Step 2: Preparing Data for Visualization
    def prepare_data_for_visualization(distances, raw_inputs):
        data = []
        for i in range(len(distances)):
            data.append([raw_inputs[i], distances[i][0]])  # distance[i][0] since baseline is all zeros and only one column
        return data

    visualization_data = prepare_data_for_visualization(distances, raw_inputs)
    print("Data prepared for visualization successfully.")

    # Step 3: Visualizing Results
    def visualize_results(visualization_data):
        inputs, cos_distances = zip(*visualization_data)
        plt.figure(figsize=(10, 6))
        plt.boxplot(cos_distances)
        plt.title('Cosine Distances of Attributions')
        plt.xlabel('Samples')
        plt.ylabel('Cosine Distance')
        plt.show()

    visualize_results(visualization_data)
    print("Visualization completed successfully.")

except Exception as e:
    computation_successful = False
    print(f"An error occurred during training or Rollout value computation: {e}")

if computation_successful:
    print("Computation and saving of Rollout values completed successfully.")
else:
    print("Computation or saving of Rollout values failed.")
