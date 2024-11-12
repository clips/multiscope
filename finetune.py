import torch
import gradio as gr
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss, classification_report
import wandb
import json
import os
import pandas as pd
from mlcm import cm
import numpy as np
import plotly.graph_objects as go
import safetensors


def matrix_to_heatmap(matrix, cmap='OrRd', colorbar_label='Value', title='Confusion Matrix', save_path=None, labels=None, annotate=True):
    """
    Converts a numpy matrix to a heatmap with annotations and tick labels using Plotly.

    Parameters:
    matrix (numpy.ndarray): The matrix to be converted to a heatmap.
    cmap (str): The colormap to use for the heatmap. Default is 'OrRd'.
    colorbar_label (str): The label for the colorbar. Default is 'Value'.
    title (str): The title for the heatmap. Default is 'Confusion Matrix'.
    save_path (str): The path to save the heatmap image. If None, the heatmap will be shown but not saved.
    labels (list): The labels for the heatmap axes. Default is None.
    annotate (bool): Whether to annotate cells with values. Default is True.

    Returns:
    fig (plotly.graph_objects.Figure): The generated plotly figure.
    """
    labels = list(labels)
    # Prepare labels for the axes
    # labels = labels if labels is not None else list(range(matrix.shape[0]))

    # Create text annotations for the heatmap
    annotations = np.round(matrix, 2).astype(str) if annotate else None

    # Create heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels + ['NPL'],  # X-axis labels
            y=labels + ['NTL'],  # Y-axis labels
            colorscale=cmap,
            text=annotations,
            hoverinfo="z",  # Show values on hover
            showscale=True,
            colorbar=dict(title=colorbar_label),
            texttemplate="%{text}" if annotate else None,  # Display annotations
            # zmin=matrix.min(),
            # zmax=matrix.max()
        )
    )

    # Set layout options
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted', tickmode='array', tickvals=list(range(len(labels) + 1)), ticktext=labels + ['NPL']),
        yaxis=dict(title='Truth', tickmode='array', tickvals=list(range(len(labels) + 1)), ticktext=labels + ['NPL']),
        autosize=False,
        width=800,
        height=600
    )

    return fig

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels if labels.any() else np.array()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        if self.labels.any():
            labels = self.labels[idx]
            encoding = {key: val.squeeze(0) for key, val in encoding.items()}
            encoding['labels'] = torch.tensor(labels, dtype=torch.float)
        return encoding

class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer, save_path):
        self.tokenizer = tokenizer
        self.save_path = save_path

    def on_save(self, args, state, control, **kwargs):
        # Save the tokenizer at each checkpoint
        checkpoint_path = f"{self.save_path}/checkpoint-{state.global_step}"
        self.tokenizer.save_pretrained(checkpoint_path)



def prepare_data(df, mlb, tokenizer, no_labels, max_length=512):
    texts = df.text
    if not no_labels:
        labels = mlb.transform(df.labels)
        dataset = CustomDataset(texts, labels, tokenizer, max_length=max_length)
    else:
        dataset = CustomDataset(texts, np.array(), tokenizer, max_length=max_length)

    return dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)) > 0.5
    
    # Micro and Macro Precision, Recall, F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Exact Match Ratio (Accuracy)
    exact_match = accuracy_score(labels, preds)
    
    # Hamming Loss
    h_loss = hamming_loss(labels, preds)
    
    # Sample-wise F1
    sample_f1 = precision_recall_fscore_support(labels, preds, average='samples')[2]

    metrics = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "exact_match_ratio": exact_match,
        "hamming_loss": h_loss,
        "sample_f1": sample_f1
    }
    
    return metrics
 

def finetune_transformer(train_df, val_df, test_df, model_name, batch_size, learning_rate, n_epochs, operations, output_dir="./results", save_best_metric="macro_f1"):
    print(operations)
    # Initialiaz MLB
    mlb = MultiLabelBinarizer()

    # Parse operations to be performed
    if 'Train' in operations and 'Test' in operations:
        print("Training and testing")
        if 'labels' in test_df.columns:
            only_predict = False
            train_labels, val_labels, test_labels = train_df.labels.tolist(), val_df.labels.tolist(), test_df.labels.tolist() 
            mlb.fit(train_labels + val_labels + test_labels)
        else:
            only_predict = True
            train_labels, val_labels = train_df.labels.tolist(), val_df.labels.tolist()
            mlb.fit(train_labels + val_labels)

        num_labels = len(mlb.classes_)

    elif 'Train' in operations and 'Test' not in operations:
        print("Only training")
        train_labels, val_labels = train_df.labels.tolist(), val_df.labels.tolist()
        mlb.fit(train_labels + val_labels)
        num_labels = len(mlb.classes_)

    else:
        print("Only testing")
        if 'labels' in test_df.columns:
            only_predict = False
            test_labels = test_df.labels.tolist() 
            mlb.fit(test_labels)
            num_labels = len(mlb.classes_)
        
        else:
            only_predict = True

    # Training a model
    if 'Train' in operations:

        # save classes to load later for inference
        with open(os.path.join(output_dir, 'classes.json'), 'w', encoding='utf8') as f:
            json.dump(list(mlb.classes_), f)

        # load model   
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError:
            raise gr.Error(f"Model '{model_name}' might not exist. Visit https://huggingface.co/models for an overview of remote models, or load a valid local model.")
        
        # create datasets
        train_dataset = prepare_data(train_df, mlb, tokenizer, no_labels=False)
        val_dataset = prepare_data(val_df, mlb, tokenizer, no_labels=False)
    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        training_args = TrainingArguments(
            learning_rate=float(learning_rate),
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            per_device_train_batch_size=int(batch_size),
            per_device_eval_batch_size=int(batch_size),
            num_train_epochs=int(n_epochs),
            load_best_model_at_end=True,
            metric_for_best_model=save_best_metric,
            save_total_limit=int(n_epochs),
            report_to='wandb'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[SaveTokenizerCallback(tokenizer, training_args.output_dir)]
        )

        # Train and evaluate
        try:
            trainer.train()
            print('Finished training!')
            
            # Evaluate on test set
            if 'Test' in operations:
                print('Testing')
                # Load best model based on metric
                best_model_path = os.path.join(output_dir, "pytorch_model.bin")
                if os.path.exists(best_model_path):
                    model = AutoModelForSequenceClassification.from_pretrained(output_dir)

                test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=True if only_predict else False)

                if only_predict:
                    trainer = Trainer(model=model)
                    test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=True)
                    logits = test_results.predictions
                    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

                    return pd.DataFrame(), pd.DataFrame(), go.Figure(), ""

                else:
                    test_results = trainer.predict(test_dataset)
                    logits = test_results.predictions
                    label_ids = test_results.label_ids
                    test_metrics = test_results.metrics
                    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

                    # Log metrics to wandb
                    wandb.log(test_metrics)

                    # Save metrics to JSON
                    metrics_json_path = os.path.join(output_dir, "metrics.json")
                    with open(metrics_json_path, 'w') as f:
                        json.dump(test_metrics, f)

                    metric_df = pd.DataFrame(data = {"metric":test_metrics.keys(), "Score": test_metrics.values()})
                    report_df = pd.DataFrame(classification_report(label_ids, preds, output_dict=True, target_names=mlb.classes_)).transpose()
                    report_df['class'] = report_df.index
                    report_df = report_df[['class', 'precision', 'recall', 'f1-score', 'support']]
                    _, cnf_matrix = cm(label_ids, preds, False)
                    cnf_matrix = matrix_to_heatmap(cnf_matrix, labels=mlb.classes_)

                    return metric_df, report_df, cnf_matrix, ""
                
            else:
                return pd.DataFrame(), pd.DataFrame(), go.Figure(), ""
            
        # Catch potential OOM error
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            message = "GPU out of memory. Try lowering the batch size or loading a smaller model!"
            return None, None, None, message
        

    # Only inference on test set using trained model
    elif 'Test' in operations and 'Train' not in operations:

        tokenizer = AutoTokenizer.from_pretrained(model_name) # model_name = fine-tuned local model in this case

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        except OSError:
            raise gr.Error(f"Model could not be loaded from '{model_name}'. Please ensure the model is available.")

        # Prepare the test dataset

        if not only_predict:

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            with open(os.path.join(output_dir, 'classes.json'), 'r', encoding='utf8') as f:
                classes = json.load(f)
            mlb = MultiLabelBinarizer(classes=classes)
            mlb.fit(test_df.labels.tolist())
            test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=False)

            # Create a Trainer instance for evaluation
            trainer = Trainer(
                model=model,
                compute_metrics=compute_metrics
             )
        

        else:  
            trainer = Trainer(
                model=model,
             )
            test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=True)
        
        # Predict test set
        test_results = trainer.predict(test_dataset)
        logits = test_results.predictions

        # Convert logits to binary predictions and save predictions
        preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
        predictions_json_path = os.path.join(output_dir, f"predictions.json")
        with open(predictions_json_path, 'w') as f:
            json.dump(preds.tolist(), f)

        label_ids = test_results.label_ids
        test_metrics = test_results.metrics

        # if labels are available, calculate metrics
        if not only_predict:
            # Log metrics and save them to JSON
            wandb.log(test_metrics)
            metrics_json_path = os.path.join(output_dir, "test_metrics.json")
            with open(metrics_json_path, 'w') as f:
                json.dump(test_metrics, f)

            # Prepare metrics DataFrame and classification report
            metric_df = pd.DataFrame(data={"metric": test_metrics.keys(), "Score": test_metrics.values()})
            report_df = pd.DataFrame(classification_report(label_ids, preds, output_dict=True, target_names=mlb.classes_)).transpose()
            report_df['class'] = report_df.index
            report_df = report_df[['class', 'precision', 'recall', 'f1-score', 'support']]

            # Generate the confusion matrix heatmap
            _, cnf_matrix = cm(label_ids, preds, False)
            cnf_matrix_fig = matrix_to_heatmap(cnf_matrix, labels=mlb.classes_)

            return metric_df, report_df, cnf_matrix_fig, ""
        
        else:
            return pd.DataFrame(), pd.DataFrame(), go.Figure(), ""




    



