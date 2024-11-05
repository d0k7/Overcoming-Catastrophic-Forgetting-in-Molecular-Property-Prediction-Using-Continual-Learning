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

# Function to calculate cosine distance between attributions and baseline
def calculate_cosine_distance(attributions):
    baseline = np.zeros_like(attributions)
    distances = cosine_distances(attributions, baseline)
    return distances

# Function to aggregate data for visualization
def prepare_data_for_visualization(attentions, tokenizer, inputs):
    decoded_inputs = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs]
    attention_values = [attention.mean(dim=1).squeeze().tolist() for attention in attentions[-1]]  # Using last layer's attentions
    
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
    plt.xlabel('XAI Attention Maps Methods')
    plt.ylabel('Cosine Distance')
    plt.show()

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
    model.save_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/GRADS_fine_tuned_bert_model")

    # Load fine-tuned model
    model = BertForSequenceClassification.from_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/GRADS_fine_tuned_bert_model")

    # Compute Grad-based attributions
    def compute_grads(model, inputs, attention_mask, labels):
        model.eval()
        inputs_dict = {
            "input_ids": inputs.to(torch.float),
            "attention_mask": attention_mask.to(torch.float)
        }
        
        labels = labels.to(torch.long)

        inputs_dict = {k: v.requires_grad_(True) for k, v in inputs_dict.items()}
        labels.requires_grad = False

        outputs = model(**inputs_dict, labels=labels)
        loss = outputs.loss
        loss.backward()

        gradients = inputs_dict["input_ids"].grad
        
        return gradients

    # Save Grad-based attributions to CSV with tokens
    def save_grads_to_csv(grads, inputs, path):
        decoded_inputs = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs]
        data = []
        for tokens, grad in zip(decoded_inputs, grads):
            for token, grad_score in zip(tokens, grad.squeeze().tolist()):
                data.append([token, grad_score])

        df = pd.DataFrame(data, columns=['Token', 'Grad Score'])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Grad-based attributions saved to {path} successfully.")
        except Exception as e:
            print(f"Error saving Grad-based attributions: {e}")

    # Loop over models and datasets for Grad-based attributions computation
    results = {}
    computation_successful = True

    for key in ["C2C"]:
        inputs, attention_mask, labels, raw_inputs = datasets[key]

        # Ensure tensors are of the correct type
        try:
            gradients = compute_grads(model, inputs, attention_mask, labels)
            save_grads_to_csv(gradients, inputs, f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_XAI_Computed/Grads_{key}.csv")
            results[key] = calculate_cosine_distance(gradients.cpu().detach().numpy())
        except Exception as e:
            print(f"An error occurred during training or Grad computation: {e}")
            computation_successful = False
            break

    if computation_successful:
        visualize_results([results[key] for key in results.keys()])

except Exception as e:
    print(f"An error occurred: {e}")
