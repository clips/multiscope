import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss, classification_report, ndcg_score
import plotly.graph_objects as go
from mlcm import cm, matrix_to_heatmap
from nltk.tokenize import word_tokenize
import wandb


class NLTKTokenizer:
    def __init__(self, language='english'):
        self.language = language

    def __call__(self, text):
        return word_tokenize(text, language=self.language)

    def set_language(self, language):
        self.language = language


def compute_metrics(labels, preds, probabilities):
    # Micro and Macro Precision, Recall, F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Exact Match Ratio (Accuracy)
    exact_match = accuracy_score(labels, preds)
    
    # Hamming Loss
    h_loss = hamming_loss(labels, preds)
    
    # Sample-wise F1
    sample_f1 = precision_recall_fscore_support(labels, preds, average='samples')[2]

    # ndcg@k
    ndcg_1 = ndcg_score(labels, probabilities, k=1)
    ndcg_3 = ndcg_score(labels, probabilities, k=3)
    ndcg_5 = ndcg_score(labels, probabilities, k=5)
    ndcg_10 = ndcg_score(labels, probabilities, k=10)

    metrics = {
        "test/micro_precision": micro_precision,
        "test/micro_recall": micro_recall,
        "test/micro_f1": micro_f1,
        "test/macro_precision": macro_precision,
        "test/macro_recall": macro_recall,
        "test/macro_f1": macro_f1,
        "test/exact_match_ratio": exact_match,
        "test/hamming_loss": h_loss,
        "test/sample_f1": sample_f1,
        "test/ndcg@1": ndcg_1,
        "test/ndcg@3": ndcg_3,
        "test/ndcg@5": ndcg_5,
        "test/ndcg@10": ndcg_10
    }
    
    return metrics


def get_most_informative_features(pipeline, class_labels, n_features=10):
    """
    Extracts the most informative features for each class from a multi-label SVM classifier wrapped in a Pipeline,
    returning a dictionary that can be easily converted to a pandas DataFrame.
    
    Parameters:
        pipeline (Pipeline): A scikit-learn Pipeline object with TfidfVectorizer and OneVsRestClassifier.
        class_labels (list): List of class labels corresponding to the target classes.
        n_features (int): Number of top features to extract for each class.
    
    Returns:
        dict: A dictionary where each entry contains 'Class', 'Feature', 'Weight', and 'Type' fields.
    """
    # Ensure the pipeline is fitted
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Pipeline should be fitted before calling this function.")
    
    # Extract components
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['svm']
    
    if not hasattr(vectorizer, 'get_feature_names_out'):
        raise ValueError("The TfidfVectorizer in the pipeline must be fitted.")
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    if not hasattr(classifier, 'estimators_'):
        raise ValueError("The classifier must be a OneVsRestClassifier with estimators_ attribute.")
    
    results = [] 
    for idx, estimator in enumerate(classifier.estimators_):
        if not hasattr(estimator, 'coef_'):
            raise ValueError("Each estimator in OneVsRestClassifier must have a coef_ attribute (e.g., linear models).")
        
        # Get coefficients for the class
        coef = estimator.coef_.toarray()[0]
        
        # Get top n_features for the current class
        top_positive_indices = np.argsort(coef)[-n_features:][::-1]
        top_negative_indices = np.argsort(coef)[:n_features]
        
        for i in top_positive_indices:
            results.append({
                'Class': class_labels[idx],
                'Feature': feature_names[i],
                'Weight': coef[i],
                'Type': 'Positive'
            })
        
        for i in top_negative_indices:
            results.append({
                'Class': class_labels[idx],
                'Feature': feature_names[i],
                'Weight': coef[i],
                'Type': 'Negative'
            })
    
    return results



def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):  # Convert NumPy floats to Python floats
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):  # Convert NumPy integers to Python ints
        return int(obj)
    return obj  # Return the object as is if no conversion is needed


def train_svm(
    output_dir, 
    train_df,
    val_df,
    test_df,
    operations,
    gridsearch_method,
    gridsearch_params,
    trained_model_path,
    stopwords,

    language='english',
    extract_features=True
    # learning_algorithm='svm',
    
):
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.concat([train_df, val_df])
    tokenizer = NLTKTokenizer(language=language)

    # label preprocessing
    mlb = MultiLabelBinarizer()
    if 'Train' in operations and 'Test' in operations:
        if 'labels' in test_df.columns:
            only_predict = False
            train_labels, test_labels = train_df.labels.tolist(), test_df.labels.tolist() 
            mlb.fit(train_labels + test_labels)

            y_test =  mlb.transform(test_df['labels']) 
            
        else:
            only_predict = True
            train_labels = train_df.labels.tolist()
            mlb.fit(train_labels)
            

        X_train = train_df['text'] 
        y_train = mlb.transform(train_df['labels'])
        X_test = test_df['text']


    elif 'Train' in operations and 'Test' not in operations:
        train_labels = train_df.labels.tolist()
        y_train = mlb.fit_transform(train_labels)
        X_train = train_df['text'] 

    else:
        if 'labels' in test_df.columns:
            only_predict = False
            test_labels = test_df.labels.tolist() 
            y_test = mlb.fit_transform(test_labels)
        
        else:
            only_predict = True

        X_test = test_df['text']
           
    # Define pipeline
    

    if 'Train' in operations:
        with open(os.path.join(output_dir, 'classes.json'), 'w', encoding='utf8') as f:
            json.dump(list(mlb.classes_), f)


        if gridsearch_method in ['Standard', 'Custom']:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords)),
                ('svm', OneVsRestClassifier(SVC(probability=True, kernel='linear')))
            ])

            msss = MultilabelStratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=42)
            grid_search = GridSearchCV(
                pipeline, gridsearch_params, cv=msss, scoring='f1_macro', n_jobs=-1, verbose=2
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            # save best params
            with open(os.path.join(output_dir, 'gridsearch_best_params.json'), 'w', encoding='utf8') as f:
                json.dump(grid_search.best_params_, f)

            # save scores per param combination
            with open(os.path.join(output_dir, 'all_gridsearch_scores.json'), 'w', encoding='utf8') as f:
                json.dump({k: make_json_serializable(v) for k,v in grid_search.cv_results_.items()}, f)
        
        else:
            # Train the model  
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2), tokenizer=tokenizer, stop_words=stopwords)),
                ('svm', OneVsRestClassifier(SVC(probability=True, kernel='linear')))
            ])
            
            pipeline.fit(X_train, y_train)
            model = pipeline

        if extract_features:
            informative_features = get_most_informative_features(model, mlb.classes_)
            feature_df = pd.DataFrame(informative_features)
            feature_df['Weight'] = feature_df['Weight'].apply(lambda x: round(x, 5))

            output_path = os.path.join(output_dir, "informative_features_per_class.json")
            with open(output_path, "w") as json_file:
                json.dump(informative_features, json_file, indent=4)
            

        # Save the trained model
        model_path = os.path.join(output_dir, 'svm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    # If only testing, load local model
    if 'Train' not in operations and 'Test' in operations:
        with open(trained_model_path, 'rb') as f:
            model = pickle.load(f)

    if 'Test' in operations:
        y_pred = model.predict(X_test)
        preds = y_pred.tolist()

        probabilities = model.predict_proba(X_test)

        with open(os.path.join(output_dir, f"predictions_svm.json"), 'w') as f:
            json.dump(preds, f)  
        
        if only_predict:
            return pd.DataFrame(), pd.DataFrame(), go.Figure(), feature_df, ""
        
        else:
            test_metrics = compute_metrics(y_test, y_pred, probabilities)

            # Log metrics to wandb
            wandb.log(test_metrics)
            wandb.finish()

            metric_df = pd.DataFrame(data={"metric": test_metrics.keys(), "Score": test_metrics.values()})
            metric_df['Score'] = metric_df['Score'].apply(lambda x: round(x, 5))
            
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, target_names=mlb.classes_)).transpose()
            report_df['class'] = report_df.index
            report_df = report_df[['class', 'precision', 'recall', 'f1-score', 'support']]
            report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].apply(lambda x: round(x, 5))

            # Generate the confusion matrix heatmap and save
            _, cnf_matrix = cm(y_test, preds, False)
            cnf_matrix_fig = matrix_to_heatmap(cnf_matrix, labels=mlb.classes_)
            cnf_matrix_fig.write_html("./visualizations/confusion_matrix.html")

            # save predictions
            with open(os.path.join(output_dir, f"predictions.json"), 'w') as f:
                json.dump(preds, f)

            # save metric df and classification report
            metric_df.to_json(os.path.join(output_dir, 'test_results.json'))
            report_df.to_json(os.path.join(output_dir, 'classification_report.json'))

            if 'Train' in operations and extract_features:
                return metric_df, report_df, cnf_matrix_fig, feature_df, ""

            else:
                 return metric_df, report_df, cnf_matrix_fig, pd.DataFrame(), ""
            
    else:
        return pd.DataFrame(), pd.DataFrame(), go.Figure(), feature_df, ""
    