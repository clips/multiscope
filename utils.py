
import pandas as pd
import numpy as np
import os
import json
import gradio as gr
from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
import plotly.graph_objects as go
from itertools import combinations
from finetune import finetune_transformer
from svm import train_svm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
import nltk


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
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(t=50),
        width=600,
        height=500
    )

    return fig


def get_label_stats(dfs, splits):
    label_stats = pd.DataFrame()
    
    for split, df in zip(splits, dfs):
        # label stats
        if 'labels' in df.columns:
            lbl_cnt_per_text = df.labels.apply(len)
            lens = df.labels.apply(len).to_numpy()

            max_cnt = max(df.labels.explode().value_counts())
            min_cnt = min(df.labels.explode().value_counts())
            mean_cnt = round(lbl_cnt_per_text.mean(),5)
            median_cnt = np.median(lens)
            std_cnt = round(lens.std(),5)

            q1_cnt = np.quantile(lbl_cnt_per_text, .25)
            q3_cnt = np.quantile(lbl_cnt_per_text, .75)

            label_stats[split] = {
                        'min count': min_cnt, 'max count': max_cnt,
                        'mean labels per text': mean_cnt, 'median labels per text': median_cnt, 
                        'std': std_cnt,
                        'Q1': q1_cnt, 'Q3': q3_cnt}
            
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
        mean_tkn_cnt = round(token_counts.mean(), 5)
        median_tkn_cnt = np.median(token_counts)
        std_tkn_cnt = round(token_counts.std(),5)
        q1_tkn_cnt = np.quantile(token_counts, .25)
        q3_tkn_cnt = np.quantile(token_counts, .75)

        token_stats[split] = {
                    'min count': min_tkn_cnt, 'max count': max_tkn_cnt,
                    'mean count': mean_tkn_cnt, 'median count': median_tkn_cnt, 'std': std_tkn_cnt,
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
            if subset == '':
                dataset = load_dataset(dataset_path, streaming=True)
            else:
                dataset = load_dataset(dataset_path, subset, streaming=True)
            
            
        except:
            raise gr.Error("Please enter a valid path to the dataset! Consult https://huggingface.co/datasets for a list of available datasets.")

        return dataset
    else:
        raise gr.Error("Please enter the path to the dataset.")



def load_local_dataset(dataset_path, split):
    if dataset_path:
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                df = df[df['subset']==split]
                df = df.drop('subset', axis=1)

                if split == 'test':
                    if df.labels.isnull().all():
                        df = df.drop('labels', axis=1) # check if labels are nan
                    else:
                        df.labels = df.labels.apply(literal_eval) # if labels are present, literal eval   
                else:
                    df.labels = df.labels.apply(literal_eval)              
            
            elif dataset_path.endswith('.xlsx'):
                df = pd.read_excel(dataset_path)
                df = df[df['subset']==split]
                df = df.drop('subset', axis=1)

                if split == 'test':
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
                        if split in data['data']:
                            df = pd.DataFrame(data['data'][split])
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


def split_data(train_df, column_name, test_size):
    train_df = train_df.reset_index()
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
    mlb = MultiLabelBinarizer()

    X = train_df['text'].values
    y = mlb.fit_transform(train_df[column_name].values)

    for train_index, val_index in msss.split(X, y):
        val_df = train_df.loc[val_index]
        new_train_df = train_df.loc[train_index]

    return new_train_df, val_df   

# Data loading function
def load_data(dataset_source, dataset_path, dataset_subset, text_column_name, label_column_name, operations, test_portion):
    if not operations:
        raise gr.Error("Please select 'Train', 'Test' or both. Please refresh the app to continue.")

    if dataset_source == "HuggingFace":
        if 'Train' in operations:
            dataset = load_huggingface_dataset(dataset_path,  dataset_subset)
            train_df = pd.DataFrame.from_dict(dataset['train'])

            train_df['text'] = train_df[text_column_name]
            train_df['labels'] = train_df[label_column_name]

            if 'val' not in dataset.keys():
                try:
                    val_df = pd.DataFrame.from_dict(dataset['validation'])
                    val_df.labels = val_df[label_column_name]
                    val_df.text = val_df['text']

                except KeyError:
                    train_df, val_df = split_data(train_df, label_column_name, test_size=test_portion)
                
            elif ('val' and 'validation' not in dataset.keys()) or 'Split Training Data' in operations:
                train_df, val_df = split_data(train_df, label_column_name, test_size=test_portion)

            train_df.labels = train_df.labels.apply(lambda x: [str(l) for l in x])
            val_df.labels = val_df.labels.apply(lambda x: [str(l) for l in x])

        if 'Test' in operations:
            test_df = load_huggingface_dataset(dataset_path,  dataset_subset)
            test_df = pd.DataFrame.from_dict(dataset['test'])
            
            test_df.text = test_df[text_column_name]
            test_df.labels = test_df[label_column_name].apply(lambda x: [str(l) for l in x])

        else:
            test_df = pd.DataFrame()
    

    else:
        label_column_name = 'labels'

        if 'Train' in operations:
            train_df = load_local_dataset(dataset_path, split="train")
            if 'Split Training Data' in operations:
                train_df, val_df = split_data(train_df, label_column_name, test_size=test_portion)
            else:
                val_df = load_local_dataset(dataset_path, split="val")
        if 'Test' in operations:
            test_df = load_local_dataset(dataset_path, split="test")
        if 'Train' not in operations:
            train_df = pd.DataFrame()
        if 'Test' not in operations:
            test_df = pd.DataFrame()


    if 'Train' in operations:
        # create label counts plot
        label_counts = train_df['labels'].explode().value_counts().plot.bar()
        label_counts.update_layout(
            showlegend=False,
            xaxis_type='category',  
            margin=dict(t=50),
            width=500,
            height=500)

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

            gr.update(value=train_df.head(10), label="Train Dataset"),

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
                gr.update(value=test_df, visible=False), # make test df visible

                gr.update(value=test_df.head(10), label="Test Dataset"),

                gr.update(value=label_stats if 'labels' in test_df.columns else pd.DataFrame(), visible= True if 'labels' in test_df.columns else False),
                token_stats, 
                gr.update(value=label_counts if 'labels' in test_df.columns else pd.DataFrame() , visible=True if 'labels' in test_df.columns else False),
                gr.update(value=correlation_matrix if 'labels' in test_df.columns else go.Figure(), visible=True if 'labels' in test_df.columns else False))
    

def generate_run_name(clf_method, gridsearch_method, model_name, batch_size, max_length, learning_rate):
    """
    Generate a unique run name for wandb based on provided arguments.
    Args:
        *args: Variable length argument list for text inputs.
    Returns:
        A string representing the run name.
    """
    # Combine all inputs into a single string, separated by underscores
    if clf_method == "Fine-tune Transformer":
        if '/' in model_name:
            model_name = model_name.split('/')[1]
        run_name = f'{model_name}_{batch_size}_{max_length}_{learning_rate}'
    
    else:
        run_name = f'SVM_{gridsearch_method}'

    return run_name

def generate_output_dir(output_dir, clf_method, model_name, batch_size, max_length, learning_rate, gridsearch_method):
    if output_dir != '':
        return output_dir
    
    else:
        output_dir = generate_run_name(clf_method, gridsearch_method, model_name, batch_size,  max_length, learning_rate)
        return output_dir
        

def generate_gridsearch_params(
    gridsearch_method,
    gs_ngram_range, n_ngram_range,
    gs_min_df, n_min_df,
    gs_max_df, n_max_df,
    gs_svm_c, n_svm_c,
    gs_svm_max_iter, n_svm_max_iter
):
   
    param_grid = {}
    if gridsearch_method =='Standard':
        param_grid['tfidf__ngram_range'] = [(1, 1), (1, 2), (1, 3)]
        param_grid['tfidf__min_df'] = [1, 2, 10]
        param_grid['tfidf__max_df'] = [0.8, 1.0]
        param_grid['svm__estimator__C'] =  [1, 0.1, 10]
        param_grid['svm__estimator__max_iter'] = [100, 500, 1000]
        


    if gridsearch_method =='Custom':
        # Parameters for TfidfVectorizer
        if gs_ngram_range:
            ngram_range_options = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]
            param_grid['tfidf__ngram_range'] = ngram_range_options[:n_ngram_range]

        if gs_min_df:
            min_df_options = [1, 2, 5, 10, 20]
            param_grid['tfidf__min_df'] = min_df_options[:n_min_df]

        if gs_max_df:
            max_df_options = [0.8, 0.9, 1.0, 0.7, 0.6]
            param_grid['tfidf__max_df'] = max_df_options[:n_max_df]

        # Parameters for SVM
        if gs_svm_c:
            svm_c_options = [1, 0.1, 10, 100, 0.01]
            param_grid['svm__estimator__C'] = svm_c_options[:n_svm_c]

        if gs_svm_max_iter:
            svm_max_iter_options = [1000, 2000, 500, 100, 5000]
            param_grid['svm__estimator__max_iter'] = svm_max_iter_options[:n_svm_max_iter]

        # Filter out any parameters that are empty or None
        param_grid = {key: value for key, value in param_grid.items() if value}

    return param_grid


def get_stopwords(stopwords_path, language):
    if stopwords_path:
        with open(stopwords_path, 'r', encoding='utf8') as f:
            stopwords = f.readlines()
    else:
        stopwords = nltk.corpus.stopwords.words(language if language else 'english')
        
    return stopwords



# Model training function
def train_model(output_dir, clf_method, model_name, train_df, val_df, test_df, batch_size, max_length, learning_rate, n_epochs, operations, gridsearch_method, trained_model_path, stopwords_path, data_language,
                # gridsearch
                gs_ngram_range, n_ngram_range, gs_min_df, n_min_df, gs_max_df, n_max_df, gs_svm_c, n_svm_c, gs_svm_max_iter, n_svm_max_iter          
                ):

    output_dir = generate_output_dir(output_dir, clf_method, model_name, batch_size, max_length, learning_rate, gridsearch_method)
    
    if clf_method == "Fine-tune Transformer":
        metric_df, report_df, cnf_matrix, feature_df, error_message = finetune_transformer(output_dir, train_df, val_df, test_df, model_name, batch_size, max_length, learning_rate, n_epochs, operations)
        return metric_df, report_df, cnf_matrix, feature_df, error_message
    
    else:
        gridsearch_params = generate_gridsearch_params(gridsearch_method, gs_ngram_range, n_ngram_range, gs_min_df, n_min_df, 
                                                       gs_max_df, n_max_df, gs_svm_c, n_svm_c, gs_svm_max_iter, n_svm_max_iter)
        stopwords = get_stopwords(stopwords_path, data_language)
        metric_df, report_df, cnf_matrix, feature_df, error_message = train_svm(output_dir, train_df, val_df, test_df, operations, gridsearch_method, gridsearch_params, trained_model_path, stopwords)
        return metric_df, report_df, cnf_matrix, feature_df, error_message



