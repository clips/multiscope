
import pandas as pd
import numpy as np
import os
import torch
import json
import gradio as gr
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer, TrainerCallback
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
import plotly.graph_objects as go
from itertools import combinations
from finetune import finetune_transformer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval


def get_token_counts(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_counts = [len(tokenizer.tokenize(text)) for text in texts]
    return np.array(token_counts)



def create_cooccurrence_matrix(column_with_lists, cmap='OrRd'):
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
        colorscale=cmap,
        colorbar=dict(title="Normalized Co-occurrence")
    ))

    return fig


def get_label_stats(dfs, splits):
    label_stats = pd.DataFrame()
    
    for split, df in zip(splits, dfs):
        # label stats
        if 'labels' in df.columns:
            lbl_cnt_per_text = df.labels.apply(len)

            max_cnt = max(df.labels.explode().value_counts())
            min_cnt = min(df.labels.explode().value_counts())
            mean_cnt = lbl_cnt_per_text.mean()
            median_cnt = np.median(df.labels.apply(len).to_numpy())

            q1_cnt = np.quantile(lbl_cnt_per_text, .25)
            q3_cnt = np.quantile(lbl_cnt_per_text, .75)

            label_stats[split] = {'min count': min_cnt, 'max count': max_cnt,
                        'mean labels per text': mean_cnt, 'median labels per text': median_cnt, 'Q1': q1_cnt, 'Q3': q3_cnt}
            
        else:
            splits.remove(split)
  
    #create column for results, since Gradio does not display df idx
    label_stats[' '] = label_stats.index

    label_stats = label_stats[[' '] + splits]

    return label_stats



def get_token_stats(dfs, splits):
    token_stats = pd.DataFrame()
    
    for split, df in zip(splits, dfs):
        token_counts = get_token_counts(df.text.tolist())
        max_tkn_cnt = max(token_counts)
        min_tkn_cnt = min(token_counts)
        mean_tkn_cnt = token_counts.mean()
        median_tkn_cnt = np.median(token_counts)
        q1_tkn_cnt = np.quantile(token_counts, .25)
        q3_tkn_cnt = np.quantile(token_counts, .75)

        token_stats[split] = {'min count': min_tkn_cnt, 'max count': max_tkn_cnt,
                    'mean count': mean_tkn_cnt, 'median count': median_tkn_cnt, 
                    'Q1': q1_tkn_cnt, 'Q3': q3_tkn_cnt}
        
    #create column for results, since Gradio does not display df idx
    token_stats[' '] = token_stats.index

    # re-order columns
    token_stats = token_stats[[' '] + splits]

    return token_stats


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
                df = df[df['subset']==subset]
                df = df.drop('subset', axis=1)
                if subset == 'test':
                    if df.labels.isnull().all():
                        df = df.drop('labels', axis=1) # check if labels are nan
                    else:
                        df.labels = df.labels.apply(literal_eval) # if labels are present, literal eval   
                else:
                    df.labels = df.labels.apply(literal_eval)              
            
            elif dataset_path.endswith('.xlsx'):
                df = pd.read_excel(dataset_path)
                df = df[df['subset']==subset]
                df = df.drop('subset', axis=1)
                if subset == 'test':
                    if df.labels.isnull().all():
                        df = df.drop('labels', axis=1) # check if labels are nan
                    else:
                        df.labels = df.labels.apply(literal_eval) # if labels are present, literal eval 
                else:
                    df.labels = df.labels.apply(literal_eval)    

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
            raise gr.Error(f"Local file with path '{dataset_path}' not found!")
        
        return df
        
    else:
        raise gr.Error("Please enter a path to the dataset.")


def split_data(train_df):
    train_df = train_df.reset_index()
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.15, random_state=0)
    mlb = MultiLabelBinarizer()

    X = train_df['text'].values
    y = mlb.fit_transform(train_df['labels'].values)

    for train_index, val_index in msss.split(X, y):
        val_df = train_df.loc[val_index]
        new_train_df = train_df.loc[train_index]

    return new_train_df, val_df   

# Data loading function
def load_data(dataset_source, dataset_path, operations):
    if not operations:
        raise gr.Error("Please select 'Train', 'Test' or both. Please refresh the app to continue.")

    if dataset_source == "HuggingFace":
        if 'Train' in operations:
            train_df = load_huggingface_dataset(dataset_path, subset="train")
            if 'Split Training Data' in operations:
                train_df, val_df = split_data(train_df)
            else:
                val_df = load_huggingface_dataset(dataset_path, subset="val")
        if 'Test' in operations:
            test_df = load_huggingface_dataset(dataset_path, subset="test")

    else:
        if 'Train' in operations:
            train_df = load_local_dataset(dataset_path, subset="train")
            if 'Split Training Data' in operations:
                train_df, val_df = split_data(train_df)
            else:
                val_df = load_local_dataset(dataset_path, subset="val")
        if 'Test' in operations:
            test_df = load_local_dataset(dataset_path, subset="test")
        if 'Train' not in operations:
            train_df = pd.DataFrame()
        if 'Test' not in operations:
            test_df = pd.DataFrame()


    if 'Train' in operations:
        # create label counts plot
        label_counts = train_df['labels'].explode().value_counts().plot.bar()
        label_counts.update_layout(xaxis_type='category')

        # label co-occurrence matrix
        correlation_matrix = create_cooccurrence_matrix(train_df.labels)

        #dataset stats
        datasets = [train_df, val_df] if test_df.empty else [train_df, val_df, test_df]
        splits = ['train', 'val'] if test_df.empty else ['train', 'val', 'test']
        label_stats = get_label_stats(datasets, splits)
        token_stats = get_token_stats(datasets, splits)

        try:
            os.mkdir("./visualizations")
        except FileExistsError:
            pass
        
        # save figures
        label_counts.write_html("./visualizations/label_counts.html")
        correlation_matrix.write_html("./visualizations/cooc_matrix.html")

        return (
            train_df,
            val_df, 
            test_df,
            label_stats,
            token_stats,
            gr.update(value=label_counts, visible=True),
            gr.update(value=correlation_matrix, visible=True)
        )
    
    # only test 
    elif 'Test' in operations:
        label_stats = get_label_stats([test_df], ['test'])
        token_stats = get_token_stats([test_df], ['test'])

        if 'labels' in test_df.columns:
            label_counts = test_df['labels'].explode().value_counts().plot.bar(title='Class counts')
            label_counts.update_layout(xaxis_type='category')
            correlation_matrix = create_cooccurrence_matrix(test_df.labels)

        return (gr.update(visible=False), # make train_df invisible in app
                pd.DataFrame(), 
                gr.update(value=test_df, visible=True), # make test df visible
                gr.update(value=label_stats if 'labels' in test_df.columns else pd.DataFrame(), visible= True if 'labels' in test_df.columns else False),
                token_stats, 
                gr.update(value=label_counts if 'labels' in test_df.columns else pd.DataFrame() , visible=True if 'labels' in test_df.columns else False),
                gr.update(value=correlation_matrix if 'labels' in test_df.columns else go.Figure(), visible=True if 'labels' in test_df.columns else False))



# Model training function
def train_model(clf_method, model_name, train_df, val_df, test_df, batch_size, learning_rate, n_epochs, operations):
    if clf_method == "Fine-tune":
        metric_df, report_df, cnf_matrix, error_message = finetune_transformer(train_df, val_df, test_df, model_name, batch_size, learning_rate, n_epochs, operations)
        return metric_df, report_df, cnf_matrix, error_message
    
    elif clf_method == "Prompt LLM":
        return f"Classifying data with {model_name} using Prompt LLM..."

    elif clf_method == "Distance-based Classification":
        return f"Classifying data with {model_name} using Distance-based Classification..."


# Gradio functions

def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe, visible=False)

def toggle_parameter_visibility(choice):
    if choice == 'Fine-tune':
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) 
    
def show_error(msg):
    if msg:
        print('triggered error msg')
        raise(gr.Error(msg))

def update_button_text(operations):
    if "Train" in operations and "Test" in operations:
        return "Train Model and Predict Test Set"
    elif "Train" in operations:
        return "Train Model"
    elif "Test" in operations:
        return "Predict Test Set"
    else:
        return "Run"

def toggle_hyperparameters(operations):
    if "Test" in operations and "Train" not in operations:
        return gr.update(value='N/A', interactive=False), gr.update(value='N/A', interactive=False)
    else:
        return gr.update(value=5, interactive=True), gr.update(value=5e-5, interactive=True)
    
def toggle_data_display(operations):
    if "Test" in operations and "Train" not in operations:
        return (gr.update(label="Test Data"), 
                gr.update(visible=False),  # Hide label stats
                gr.update(visible=False),  # Hide class counts plot
                gr.update(visible=False))  # Hide co-occurrence matrix
    else:
        return (gr.update(label="Training Dataset"),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True))

def toggle_classification_results(operations):
    if "Train" in operations and "Test" not in operations:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)