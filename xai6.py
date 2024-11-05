import os
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

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
    model.save_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/AttGrads_fine_tuned_bert_model")

    # Load fine-tuned model
    model = BertForSequenceClassification.from_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/AttGrads_fine_tuned_bert_model")

    # Compute Attention-based Gradients (AttGrads)
    def compute_att_grads(model, inputs, attention_mask):
        model.eval()
        inputs_dict = {
            "input_ids": inputs,
            "attention_mask": attention_mask
        }
        outputs = model(**inputs_dict, output_attentions=True)
        attentions = outputs.attentions  # List of attention maps for each layer
        logits = outputs.logits
        target_class = logits.argmax(dim=1)
        target_logits = logits[range(len(logits)), target_class]
        
        # Compute gradients with respect to the attention scores
        grads = torch.autograd.grad(target_logits.sum(), attentions[-1], retain_graph=True)[0]  # Gradients of the last layer attentions
        
        return grads

    # Save Attention-based Gradients to CSV with tokens
    def save_att_grads_to_csv(grads, inputs, path):
        decoded_inputs = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs]
        data = []
        for tokens, grad in zip(decoded_inputs, grads):
            for token, gradient in zip(tokens, grad.mean(dim=1).squeeze().tolist()):  # Averaging gradients across heads
                data.append([token, gradient])

        df = pd.DataFrame(data, columns=['Token', 'Gradient Score'])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Attention-based gradients saved to {path} successfully.")
        except Exception as e:
            print(f"Error saving attention-based gradients: {e}")

    # Compute Cosine Distance
    def compute_cosine_distance(grad, baseline=None):
        if baseline is None:
            baseline = np.zeros_like(grad)
        grad = grad.flatten().reshape(1, -1)
        baseline = baseline.flatten().reshape(1, -1)
        return cosine_distances(grad, baseline)[0][0]

    # Save Cosine Distances to CSV
    def save_cosine_distances_to_csv(distances, path):
        df = pd.DataFrame(distances, columns=['Cosine Distance'])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Cosine distances saved to {path} successfully.")
        except Exception as e:
            print(f"Error saving cosine distances: {e}")

    # Loop over models and datasets for Attention-based Gradients computation
    results = {}
    computation_successful = True

    for key in ["C2C"]:
        inputs, attention_mask, labels, raw_inputs = datasets[key]

        inputs = torch.tensor(inputs, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        grads = compute_att_grads(model, inputs, attention_mask)

        # Calculate cosine distances
        cosine_distances = [compute_cosine_distance(grad) for grad in grads]

        # Save attention-based gradients
        csv_path_grads = f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_XAI_Computed/{key}_AttGrads.csv"
        save_att_grads_to_csv(grads, inputs, csv_path_grads)

        # Save cosine distances
        csv_path_distances = f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_XAI_Computed/{key}_Cosine_Distances.csv"
        save_cosine_distances_to_csv(cosine_distances, csv_path_distances)

        results[key] = grads

    # Visualization of cosine distances
    def visualize_cosine_distances(distances, title="Cosine Distances Box Plot"):
        plt.figure(figsize=(10, 6))
        plt.boxplot(distances)
        plt.title(title)
        plt.xlabel('XAI Methods')
        plt.ylabel('Cosine Distance')
        plt.xticks([1], ['AttGrads'])  # Assuming only AttGrads for now
        plt.show()

    cosine_distances_all = []
    for key in ["C2C"]:
        csv_path_distances = f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_XAI_Computed/{key}_Cosine_Distances.csv"
        if os.path.exists(csv_path_distances):
            df = pd.read_csv(csv_path_distances)
            cosine_distances = df['Cosine Distance'].tolist()
            cosine_distances_all.append(cosine_distances)

    # Visualize Cosine Distances using a Box Plot
    visualize_cosine_distances(cosine_distances_all, title="Cosine Distances Box Plot")

except Exception as e:
    computation_successful = False
    print(f"An error occurred during training or attention-based gradients computation: {e}")

if computation_successful:
    print("Computation and saving of attention-based gradients and cosine distances completed successfully.")
else:
    print("Computation or saving of attention-based gradients and cosine distances failed.")
 