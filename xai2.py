import os
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import shap

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
    model.save_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/SHAP_fine_tuned_bert_model")

    # Load fine-tuned model
    model = BertForSequenceClassification.from_pretrained("/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/All_Finedtune_Model_4_XAI/SHAP_fine_tuned_bert_model")

    # Compute SHAP values
    def compute_shap_values(model, dataset, tokenizer):
        model.eval()
        
        class ModelWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
            
            def predict(self, texts):
                inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return outputs.logits.detach().numpy()
        
        wrapped_model = ModelWrapper(model, tokenizer)
        explainer = shap.Explainer(wrapped_model.predict, dataset)
        shap_values = explainer(dataset)
        return shap_values

    # Save SHAP values to CSV
    def save_shap_values_to_csv(shap_values, raw_inputs, path):
        data = []
        for i in range(len(shap_values)):
            input_text = raw_inputs[i]
            shap_value = shap_values[i].values
            data.append([input_text] + shap_value.tolist())
        
        df = pd.DataFrame(data, columns=['Input Text'] + [f'SHAP Value {i}' for i in range(len(shap_values[0]))])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"SHAP values saved to {path} successfully.")
        except Exception as e:
            print(f"Error saving SHAP values: {e}")

    # Loop over models and datasets for SHAP computation
    results = {}
    computation_successful = True

    for key in ["C2C"]:
        inputs, attention_mask, labels, raw_inputs = datasets[key]

        inputs = inputs.tolist()
        shap_values = compute_shap_values(model, inputs, tokenizer)

        csv_path = f"/scratch/sakshi.rs.cse21.itbhu/dheeraj/XAI/XAI1/{key}_SHAP_Values.csv"
        save_shap_values_to_csv(shap_values, raw_inputs, csv_path)

        results[key] = shap_values

except Exception as e:
    computation_successful = False
    print(f"An error occurred during training or SHAP value computation: {e}")

if computation_successful:
    print("Computation and saving of SHAP values completed successfully.")
else:
    print("Computation or saving of SHAP values failed.")



{'loss': 0.3983, 'grad_norm': 8.076696395874023, 'learning_rate': 0.0, 'epoch': 3.0}
{'eval_loss': 0.33976438641548157, 'eval_runtime': 13.6359, 'eval_samples_per_second': 14.961, 'eval_steps_per_second': 1.907, 'epoch': 3.0}
{'train_runtime': 3589.4515, 'train_samples_per_second': 1.534, 'train_steps_per_second': 0.192, 'train_loss': 0.4538692757703256, 'epoch': 3.0}
