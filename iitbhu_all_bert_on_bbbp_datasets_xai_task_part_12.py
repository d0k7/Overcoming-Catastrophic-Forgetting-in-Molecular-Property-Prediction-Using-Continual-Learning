# -*- coding: utf-8 -*-
"""IITBHU ALL BERT ON BBBP DATASETS XAI Task Part 12

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sTGhDIFegcr2d3jF1uXhyY7M5hQMuj94

**1) XAI FOR BERT ON UNMASKED BBBP DATASETS**

**Step 1: Setup and Loading the Data**
"""

# Install required libraries
!pip install transformers captum shap lime

import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from google.colab import drive
import captum.attr
import shap
import lime
import numpy as np

# Mount Google Drive
#drive.mount('/content/drive')

# Define model and dataset paths
model_paths = {
    "C2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/C2C_BERT_Model",
    "R2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/R2C_BERT_Model",
    "E2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/E2C_BERT_Model"
}

dataset_paths = {
    "C2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBP C2C.csv",
    "R2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBPR2C.csv",
    "E2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBPE2C.csv"
}

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to load and tokenize datasets
def load_and_tokenize_dataset(path):
    df = pd.read_csv(path)
    inputs = df['Canonical_SMILES'].tolist()
    labels = df['Label'].tolist()
    encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
    return encoded_inputs['input_ids'].long(), encoded_inputs['attention_mask'].long(), torch.tensor(labels).long()

# Load the model and dataset you want to process
model_key = "C2C"  # Change this to the model you want to process: "C2C", "R2C", "E2C"
model = BertForSequenceClassification.from_pretrained(model_paths[model_key])
input_data, attention_mask, labels = load_and_tokenize_dataset(dataset_paths[model_key])

"""**Step 2: Define XAI Methods**"""

# Define XAI methods
def compute_integrated_gradients(model, inputs, labels, attention_mask):
    def forward_func(input_ids, attention_mask):
        return model(input_ids.long(), attention_mask=attention_mask.long()).logits
    ig = captum.attr.IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(inputs.long(), target=labels, additional_forward_args=(attention_mask.long(),), return_convergence_delta=True)
    return attributions

"""**Step 3: Apply Integrated Gradients in Batches**"""

from torch.utils.data import DataLoader, TensorDataset

# Create a DataLoader for batching
batch_size = 4  # Adjust based on your available memory
dataset = TensorDataset(input_data, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Function to process batches
def process_batches(model, dataloader):
    all_attributions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, label_batch = batch
            attributions_ig = compute_integrated_gradients(model, input_ids, label_batch, attention_masks)
            all_attributions.append(attributions_ig)
    return torch.cat(all_attributions)

# Process and get attributions
attributions_ig = process_batches(model, dataloader)

# Print or save the results as needed
print("Integrated Gradients Attributions:", attributions_ig)

# Install required libraries
!pip install transformers captum shap lime

import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from google.colab import drive
import captum.attr
import shap
import lime
import numpy as np

# Mount Google Drive
#drive.mount('/content/drive')

# Define model and dataset paths
model_paths = {
    "C2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/C2C_BERT_Model",
    #"R2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/R2C_BERT_Model",
    #"E2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/E2C_BERT_Model"
}

dataset_paths = {
    "C2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBP C2C.csv",
    #"R2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBPR2C.csv",
    #"E2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBPE2C.csv"
}

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to load and tokenize datasets
def load_and_tokenize_dataset(path):
    df = pd.read_csv(path)
    inputs = df['Canonical_SMILES'].tolist()
    labels = df['Label'].tolist()
    encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
    return encoded_inputs['input_ids'].long(), encoded_inputs['attention_mask'].long(), torch.tensor(labels).long()

# Load datasets
datasets = {key: load_and_tokenize_dataset(path) for key, path in dataset_paths.items()}

# Load models
models = {key: BertForSequenceClassification.from_pretrained(path) for key, path in model_paths.items()}

# Define XAI methods
def compute_integrated_gradients(model, inputs, labels, attention_mask):
    def forward_func(input_ids, attention_mask):
        return model(input_ids.long(), attention_mask=attention_mask.long()).logits
    ig = captum.attr.IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(inputs.long(), target=labels, additional_forward_args=(attention_mask.long(),), return_convergence_delta=True)
    return attributions

def compute_shap_values(model, inputs, attention_mask):
    def forward_func(input_ids):
        return model(input_ids.long(), attention_mask=attention_mask.long()).logits
    explainer = shap.Explainer(forward_func, inputs.long())
    shap_values = explainer(inputs.long())
    return shap_values

def get_attention_maps(model, inputs, attention_mask):
    outputs = model(inputs.long(), attention_mask=attention_mask.long(), output_attentions=True)
    return outputs.attentions

def attention_rollout(attentions):
    rollout = attentions[0]
    for attn in attentions[1:]:
        rollout = torch.matmul(rollout, attn)
    return rollout.mean(dim=1)

def compute_gradients(model, inputs, labels, attention_mask):
    inputs = inputs.long()
    inputs.requires_grad = True
    outputs = model(inputs, attention_mask=attention_mask.long())
    loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
    loss.backward()
    gradients = inputs.grad
    return gradients

def compute_attention_gradients(model, inputs, labels, attention_mask):
    outputs = model(inputs.long(), attention_mask=attention_mask.long(), output_attentions=True)
    attentions = outputs.attentions[-1]
    attentions.retain_grad()
    loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
    loss.backward()
    attention_gradients = attentions.grad
    return attention_gradients

def compute_contextual_attention(model, inputs, attention_mask, context_vector):
    outputs = model(inputs.long(), attention_mask=attention_mask.long())
    hidden_states = outputs.hidden_states[-1]
    contextual_attention = torch.matmul(hidden_states, context_vector)
    return contextual_attention

def compute_attention_cat(model, inputs, attention_mask, context_vector):
    outputs = model(inputs.long(), attention_mask=attention_mask.long(), output_attentions=True)
    attentions = outputs.attentions[-1]
    hidden_states = outputs.hidden_states[-1]
    contextual_attention = torch.matmul(hidden_states, context_vector)
    attention_cat = attentions * contextual_attention.unsqueeze(-1)
    return attention_cat

def compute_lime_explanations(model, inputs, attention_mask):
    def predict(inputs):
        inputs = torch.tensor(tokenizer(inputs, padding='max_length', truncation=True, max_length=50, return_tensors='pt')['input_ids'].long())
        attention_mask = torch.tensor(tokenizer(inputs, padding='max_length', truncation=True, max_length=50, return_tensors='pt')['attention_mask'].long())
        return model(inputs.long(), attention_mask=attention_mask.long()).logits.detach().numpy()
    explainer = lime.lime_text.LimeTextExplainer()
    explanations = [explainer.explain_instance(text, predict) for text in tokenizer.batch_decode(inputs, skip_special_tokens=True)]
    return explanations

# Example: Apply all XAI methods to all models and datasets
results = {}

for key in models.keys():
    model = models[key]
    input_data, attention_mask, labels = datasets[key]

    # Ensure input tensors are LongTensor
    input_data = input_data.long()
    attention_mask = attention_mask.long()
    labels = labels.long()

    # Create a context vector for each model
    context_vector = torch.randn(model.config.hidden_size)  # Example context vector

    attributions_ig = compute_integrated_gradients(model, input_data, labels, attention_mask)
    shap_values = compute_shap_values(model, input_data, attention_mask)
    attention_maps = get_attention_maps(model, input_data, attention_mask)
    attention_rollout_map = attention_rollout(attention_maps)
    gradients = compute_gradients(model, input_data, labels, attention_mask)
    attention_gradients = compute_attention_gradients(model, input_data, labels, attention_mask)
    cat = compute_contextual_attention(model, input_data, attention_mask, context_vector)
    att_cat = compute_attention_cat(model, input_data, attention_mask, context_vector)
    lime_explanations = compute_lime_explanations(model, input_data, attention_mask)

    results[key] = {
        'IG': attributions_ig,
        'SHAP': shap_values,
        'Attention Maps': attention_maps,
        'Attention Rollout': attention_rollout_map,
        'Gradients': gradients,
        'Attention Gradients': attention_gradients,
        'CAT': cat,
        'AttCAT': att_cat,
        'LIME': lime_explanations
    }

# Print or save the results as needed
print(results)

import pandas as pd

# Load the CSV file into a DataFrame with a specified encoding
file_path = "/content/drive/MyDrive/rand/Bitter M C2C.csv"

# Try different encodings if the first one doesn't work
encodings = ['latin1', 'iso-8859-1', 'cp1252']

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully loaded with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Failed to load with encoding: {enc}. Error: {e}")

# If the DataFrame is successfully loaded, proceed with the rest of the code
if 'df' in locals():
    # Filter the DataFrame to include only rows where Label is 1
    df_filtered = df[df['Label'] == 1]

    # Count the occurrences of each unique value in the Taste column
    taste_counts = df_filtered['Taste'].value_counts()

    # Print the counts
    print(taste_counts)
else:
    print("Failed to load the CSV file with the tried encodings.")

import pandas as pd

# Load the CSV file into a DataFrame with a specified encoding
file_path = "/content/drive/MyDrive/rand/Bitter M C2C.csv"

# Try different encodings if the first one doesn't work
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully loaded with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Failed to load with encoding: {enc}. Error: {e}")

# If the DataFrame is successfully loaded, proceed with the rest of the code
if 'df' in locals():
    # Filter the DataFrame to include only rows where Label is 1
    df_filtered = df[df['Label'] == 1]

    # Filter the DataFrame to include only the desired taste categories
    desired_tastes = ['Bitter', 'Sweet', 'Tasteless']
    df_filtered = df_filtered[df_filtered['Taste'].isin(desired_tastes)]

    # Count the occurrences of each unique value in the Taste column
    taste_counts = df_filtered['Taste'].value_counts()

    # Print the counts
    print(taste_counts)
else:
    print("Failed to load the CSV file with the tried encodings.")

import pandas as pd

# Load the CSV file into a DataFrame with a specified encoding
file_path = "/content/drive/MyDrive/rand/Bitter M C2C.csv"

# Try different encodings if the first one doesn't work
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully loaded with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Failed to load with encoding: {enc}. Error: {e}")

# If the DataFrame is successfully loaded, proceed with the rest of the code
if 'df' in locals():
    # Count the occurrences of each unique value in the Taste column
    taste_counts = df['Taste'].value_counts()

    # Print the counts
    print(taste_counts)
else:
    print("Failed to load the CSV file with the tried encodings.")

import pandas as pd

# Load the CSV file into a DataFrame with a specified encoding
file_path = "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBP (4).csv"

# Try different encodings if the first one doesn't work
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully loaded with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Failed to load with encoding: {enc}. Error: {e}")

# If the DataFrame is successfully loaded, proceed with the rest of the code
if 'df' in locals():
    # Count the occurrences of each unique value in the Label column
    label_counts = df['Label'].value_counts()

    # Print the counts
    print(label_counts)
else:
    print("Failed to load the CSV file with the tried encodings.")

# Install required libraries
!pip install transformers captum shap lime

import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from google.colab import drive
import captum.attr
import shap
import lime
import numpy as np

# Mount Google Drive
#drive.mount('/content/drive')

# Define model and dataset paths
model_paths = {
    "C2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/C2C_BERT_Model",
    #"R2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/R2C_BERT_Model",
    #"E2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/Saved Model/E2C_BERT_Model"
}

dataset_paths = {
    "C2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBP C2C.csv",
    #"R2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBPR2C.csv",
    #"E2C": "/content/drive/MyDrive/Data for IIT BHU/New XAI Datasets/BBBPE2C.csv"
}

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to load and tokenize datasets
def load_and_tokenize_dataset(path):
    df = pd.read_csv(path)
    inputs = df['Canonical_SMILES'].tolist()
    labels = df['Label'].tolist()
    encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
    return encoded_inputs['input_ids'].long(), encoded_inputs['attention_mask'].long(), torch.tensor(labels).long()

# Load datasets
datasets = {key: load_and_tokenize_dataset(path) for key, path in dataset_paths.items()}

# Load models
models = {key: BertForSequenceClassification.from_pretrained(path) for key, path in model_paths.items()}

# Define XAI methods
def compute_integrated_gradients(model, inputs, labels, attention_mask):
    def forward_func(input_ids, attention_mask):
        return model(input_ids.long(), attention_mask=attention_mask.long()).logits
    ig = captum.attr.IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(inputs.long(), target=labels, additional_forward_args=(attention_mask.long(),), return_convergence_delta=True)
    return attributions

# Define other XAI methods (SHAP, LIME, etc.) here...

# Loop over each dataset and model
results = {}
for key in models.keys():
    model = models[key]
    inputs, attention_mask, labels = datasets[key]

    print(f"Processing model {key}...")

    results[key] = {}

    # Compute Integrated Gradients (already defined)
    print("Computing Integrated Gradients...")
    attributions_ig = compute_integrated_gradients(model, inputs, labels, attention_mask)
    results[key]['integrated_gradients'] = attributions_ig

    # Define and compute other XAI methods here...

# Results are stored in the `results` dictionary
print("XAI computations completed successfully.")

