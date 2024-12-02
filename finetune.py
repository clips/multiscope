import torch
import gradio as gr
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss, classification_report, ndcg_score
import wandb
import json
import os
import pandas as pd
from mlcm import cm, matrix_to_heatmap
import numpy as np
import plotly.graph_objects as go
import safetensors
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            labels = self.labels[idx]
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



def prepare_data(df, mlb, tokenizer, no_labels, max_length):
    texts = df.text
    labels = None

    if not no_labels:
        labels = mlb.transform(df.labels)
        
    dataset = CustomDataset(texts, labels=labels, tokenizer=tokenizer, max_length=max_length)

    return dataset


def compute_metrics(pred):
    labels = pred.label_ids
    sig = torch.sigmoid(torch.tensor(pred.predictions))
    preds = sig > 0.5
    
    # Micro and Macro Precision, Recall, F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Exact Match Ratio (Accuracy)
    exact_match = accuracy_score(labels, preds)
    
    # Hamming Loss
    h_loss = hamming_loss(labels, preds)

    # NDCG@k
    ndcg_1 = ndcg_score(labels, sig, k=1)
    ndcg_3 = ndcg_score(labels, sig, k=3)
    ndcg_5 = ndcg_score(labels, sig, k=5)
    ndcg_10 = ndcg_score(labels, sig, k=10)
    
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
        "sample_f1": sample_f1,

        "ndcg@1": ndcg_1,
        "ndcg@3": ndcg_3,
        "ndcg@5": ndcg_5,
        "ndcg@10": ndcg_10

    }
    
    return metrics
 


def predict(test_dataset, model, batch_size):
    predictions = []
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = {k: v.to(model.device) for k, v in batch.items()}  # Exclude 'labels' if not present
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            sig = torch.sigmoid(torch.tensor(logits))
            preds = (sig > 0.5).int().numpy()
            predictions.append(preds.tolist())

    return predictions, sig



def finetune_transformer(
        output_dir, 
        train_df, 
        val_df, 
        test_df, 
        model_name, 
        batch_size, 
        max_length,
        learning_rate, 
        n_epochs, 
        operations,
        save_best_metric="macro_f1"
    ):
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialiaz MLB
    mlb = MultiLabelBinarizer()
    batch_size = int(batch_size)

    # Parse operations to be performed
    if 'Train' in operations and 'Test' in operations:
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
        train_labels, val_labels = train_df.labels.tolist(), val_df.labels.tolist()
        mlb.fit(train_labels + val_labels)
        num_labels = len(mlb.classes_)

    else:
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
        train_dataset = prepare_data(train_df, mlb, tokenizer, no_labels=False, max_length=max_length)
        val_dataset = prepare_data(val_df, mlb, tokenizer, no_labels=False, max_length=max_length)
    
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        training_args = TrainingArguments(
            learning_rate=float(learning_rate),
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=int(n_epochs),
            load_best_model_at_end=True,
            metric_for_best_model=save_best_metric,
            save_total_limit=int(n_epochs),
            report_to='wandb',
            # run_name = f'{model_name}_{batch_size}_{learning_rate}'
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
            
            # Evaluate on test set
            if 'Test' in operations:
                # Load best model based on metric
                best_model_path = os.path.join(output_dir, "pytorch_model.bin")
                if os.path.exists(best_model_path):
                    model = AutoModelForSequenceClassification.from_pretrained(output_dir)

                test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=only_predict, max_length=max_length)

                if only_predict:
                    test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=True, max_length=max_length)
                    preds, sig = predict(test_dataset, model, batch_size)

                    with open(os.path.join(output_dir, f"predictions.json"), 'w') as f:
                        json.dump(preds, f)

                    with open(os.path.join(output_dir, f"probabilities.json"), 'w') as f:
                        json.dump(sig.tolist(), f)
            
                    return pd.DataFrame(), pd.DataFrame(), go.Figure(), pd.DataFrame(), ""

                else:
                    test_results = trainer.predict(test_dataset)
                    logits = test_results.predictions
                    label_ids = test_results.label_ids
                    test_metrics = test_results.metrics
                    sig = torch.sigmoid(torch.tensor(logits))
                    preds = (sig > 0.5).int().numpy()

                    # Log metrics to wandb
                    wandb.log(test_metrics)
                    wandb.finish()

                    # Save metrics to JSON
                    metrics_json_path = os.path.join(output_dir, "metrics.json")
                    with open(metrics_json_path, 'w') as f:
                        json.dump(test_metrics, f)

                    metric_df = pd.DataFrame(data = {"metric":test_metrics.keys(), "Score": test_metrics.values()})
                    metric_df['Score'] = metric_df['Score'].apply(lambda x: round(x, 5))

                    report_df = pd.DataFrame(classification_report(label_ids, preds, output_dict=True, target_names=mlb.classes_)).transpose()
                    report_df['class'] = report_df.index
                    report_df = report_df[['class', 'precision', 'recall', 'f1-score', 'support']]
                    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].apply(lambda x: round(x, 5))

                    # get confusion matrix and save
                    _, cnf_matrix = cm(label_ids, preds, False)
                    cnf_matrix_fig = matrix_to_heatmap(cnf_matrix, labels=mlb.classes_)
                    cnf_matrix_fig.write_html("./visualizations/confusion_matrix.html")

                    # save predictions
                    with open(os.path.join(output_dir, f"predictions.json"), 'w') as f:
                        json.dump(preds.tolist(), f)

                    with open(os.path.join(output_dir, f"probabilities.json"), 'w') as f:
                        json.dump(sig.tolist(), f)

                    # save metric df and classification report
                    metric_df.to_json(os.path.join(output_dir, 'test_results.json'))
                    report_df.to_json(os.path.join(output_dir, 'classification_report.json'))

                    return metric_df, report_df, cnf_matrix_fig, pd.DataFrame(), ""
                
            else:
                return pd.DataFrame(), pd.DataFrame(), go.Figure(), pd.DataFrame(), ""
            
        # Catch potential OOM error
        except torch.cuda.OutOfMemoryError as e:
            message = "GPU out of memory. Try lowering the batch size or loading a smaller model!"
            return None, None, None, None, message
        

    # Only inference on test set using trained model
    elif 'Test' in operations and 'Train' not in operations:
    
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name) # model_name = fine-tuned local model in this case
            model = AutoModelForSequenceClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        except OSError:
            raise gr.Error(f"Model could not be loaded from '{model_name}'. Please ensure the model is available.")

        # Prepare the test dataset
        try:
            if only_predict:
                test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=True, max_length=max_length)
                preds = predict(test_dataset, model, batch_size)

                # save predictions
                with open(os.path.join(output_dir, f"predictions.json"), 'w') as f:
                    json.dump(preds, f)

                return pd.DataFrame(), pd.DataFrame(), go.Figure(), pd.DataFrame(), ""


            # only make predictions
            else:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                with open(os.path.join(output_dir, 'classes.json'), 'r', encoding='utf8') as f:
                    classes = json.load(f)
                mlb = MultiLabelBinarizer(classes=classes)
                mlb.fit(test_df.labels.tolist())
                test_dataset = prepare_data(test_df, mlb, tokenizer, no_labels=False, max_length=max_length)

                # Create a Trainer instance for evaluation
                trainer = Trainer(
                    model=model,
                    compute_metrics=compute_metrics
                )
                
                # Predict test set
                test_results = trainer.predict(test_dataset)
                logits = test_results.predictions
                sig = torch.sigmoid(torch.tensor(logits))
                preds = (sig > 0.5).int().numpy()
                preds = [p.tolist() for p in preds]   

                label_ids = test_results.label_ids
                test_metrics = test_results.metrics

                # Log metrics and save them to JSON
                wandb.log(test_metrics)

                metrics_json_path = os.path.join(output_dir, "test_metrics.json")
                with open(metrics_json_path, 'w') as f:
                    json.dump(test_metrics, f)

                # Prepare metrics DataFrame and classification report
                metric_df = pd.DataFrame(data={"metric": test_metrics.keys(), "Score": test_metrics.values()})
                metric_df['Score'] = metric_df['Score'].apply(lambda x: round(x, 5))
                
                report_df = pd.DataFrame(classification_report(label_ids, preds, output_dict=True, target_names=mlb.classes_)).transpose()
                report_df['class'] = report_df.index
                report_df = report_df[['class', 'precision', 'recall', 'f1-score', 'support']]
                report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].apply(lambda x: round(x, 5))

                # Generate the confusion matrix heatmap and save
                _, cnf_matrix = cm(label_ids, preds, False)
                cnf_matrix_fig = matrix_to_heatmap(cnf_matrix, labels=mlb.classes_)
                cnf_matrix_fig.write_html("./visualizations/confusion_matrix.html")

                # save predictions
                with open(os.path.join(output_dir, f"predictions.json"), 'w') as f:
                    json.dump(preds, f)

                with open(os.path.join(output_dir, f"probabilities.json"), 'w') as f:
                    json.dump(sig.tolist(), f)

                # save metric df and classification report
                metric_df.to_json('./results/test_results.json')
                report_df.to_json('./results/classification_report.json')

                return metric_df, report_df, cnf_matrix_fig, pd.DataFrame(), ""
            
        # Catch potential OOM error
        except torch.cuda.OutOfMemoryError as e:
            message = "GPU out of memory. Try lowering the batch size or loading a smaller model!"
            return None, None, None, None, message