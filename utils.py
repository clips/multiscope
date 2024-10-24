
import pandas as pd
import numpy as np
import os
import torch
import json
import gradio as gr
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
import plotly.graph_objects as go
from itertools import combinations
from finetune import finetune_transformer


def create_cooccurrence_matrix(column_with_lists):
    # Flatten the list of labels and get unique labels
    unique_labels = list(set(label for sublist in column_with_lists for label in sublist))

    # Initialize an empty co-occurrence matrix
    co_occurrence_matrix = pd.DataFrame(0, index=sorted(unique_labels), columns=sorted(unique_labels))

    # Count co-occurrences
    for sublist in column_with_lists:
            for label1, label2 in combinations(sublist, 2):
                co_occurrence_matrix.loc[label1, label2] += 1
                co_occurrence_matrix.loc[label2, label1] += 1  # Ensure symmetry
            
    # Normalize the matrix by dividing each value by the maximum value in the matrix
    row_sums = co_occurrence_matrix.sum(axis=1)
    normalized_matrix = co_occurrence_matrix.div(row_sums, axis=0).fillna(0)

    # Generate the heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=normalized_matrix.to_numpy(),
        x=normalized_matrix.columns,
        y=normalized_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title="Normalized Co-occurrence")
    ))

    # Set axis labels
    fig.update_layout(
        xaxis_title="Label",
        yaxis_title="Label"
    )

    return fig





# Dataset utils
def load_huggingface_dataset(dataset_path, subset):
    if dataset_path:
        try:
            dataset = load_dataset(dataset_path, streaming=True)
            df = pd.DataFrame.from_dict(dataset[subset])
        
        except:
            raise gr.Error("Please enter a valid path to the dataset! Consult https://huggingface.co/datasets for a list of available datasets.")

        return df
    else:
        raise gr.Error("Please enter the path to the dataset.")

def load_local_dataset(dataset_path, subset):
    if dataset_path:
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            
            elif dataset_path.endswith('.xlsx'):
                df = pd.read_excel(dataset_path)

            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                    if 'data' in data:
                        if subset in data['data']:
                            df = pd.DataFrame(data['data'][subset])
                        else:
                            raise gr.Error(f"Ensure that the subset names are correct! Found {list(data.keys())}, but expected ['train', 'val', 'test']")
                        
                    else:
                        raise gr.Error("Ensure that your .json file conforms to the predefined structure.")
            else:
                raise gr.Error("Please load a .csv, .xlsx or .json file!")

        except FileNotFoundError as e:
            raise gr.Error("Local file not found!")
        
        return df.sample(n=50, random_state=42)
        
    else:
        raise gr.Error("Please enter a path to the dataset.")

    


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(labels, dtype=torch.float)
        return encoding
    

# Data loading function
def load_data(dataset_source, dataset_path):

    if dataset_source == "HuggingFace":
        train_df = load_huggingface_dataset(dataset_path, subset="train")
        val_df = load_huggingface_dataset(dataset_path, subset="val")
        test_df = load_huggingface_dataset(dataset_path, subset="test")

    else:
        train_df = load_local_dataset(dataset_path, subset="train")
        val_df = load_local_dataset(dataset_path, subset="val")
        test_df = load_local_dataset(dataset_path, subset="test")

    if not train_df.empty:
        label_counts = train_df['labels'].explode().value_counts().plot.bar(title='Class counts')
        label_counts.update_layout(xaxis_type='category')

        correlation_matrix = create_cooccurrence_matrix(train_df.labels)

        try:
            os.mkdir("./visualizations")
        except FileExistsError:
            pass

        label_counts.write_html("./visualizations/label_counts.html")
        correlation_matrix.write_html("./visualizations/cooc_matrix.html")

        # return train_df, val_df, test_df, label_counts, correlation_matrix
        return (
            train_df,
            val_df, 
            test_df,
            gr.update(value=label_counts, visible=True),
            gr.update(value=correlation_matrix, visible=True)
        )
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

# Model training function
def train_model(clf_method, model_name, train_df, val_df, test_df, batch_size, learning_rate):
    if not train_df.empty:
        if clf_method == "Fine-tune":
            metric_df, report_df, cnf_matrix, error_message  = finetune_transformer(train_df, val_df, test_df, model_name, batch_size, learning_rate)
            return metric_df, report_df, cnf_matrix, error_message
        
        elif clf_method == "Prompt LLM":
            return f"Classifying data with {model_name} using Prompt LLM..."
    
        elif clf_method == "Distance-based Classification":
            return f"Classifying data with {model_name} using Distance-based Classification..."

    else:
        return "No data loaded for training."
    

def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe, visible=False)


def toggle_parameter_visibility(choice):
    if choice == 'Fine-tune':
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    