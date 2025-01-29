import gradio as gr
import argparse
import pandas as pd
import plotly.graph_objects as go
from utils import (load_data, train_model)
from gradio_utils import (toggle_parameter_visibility,show_error, update_button_text, toggle_hyperparameters, toggle_data_display, 
                          toggle_classification_results, toggle_subset_display, toggle_set_sizes, toggle_gridsearch_params, toggle_feature_df, wandb_integration)
import wandb

pd.options.plotting.backend = "plotly"

# CSS THEMES___________________________________________________________________________________________________

css = """
h1 {
    display: block;
    text-align: center;
    font-size: 32pt;
}
.progress-bar-wrap.progress-bar-wrap.progress-bar-wrap
{
	border-radius: var(--input-radius);
	height: 1.25rem;
	margin-top: 1rem;
	overflow: hidden;
	width: 70%;

"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="amber",
).set(
    button_primary_background_fill='*secondary_500',
    button_primary_background_fill_hover='*secondary_400',
    block_label_background_fill='*primary_50',
)


#APP INTERFACE______________________________________________________________________________________________

# Gradio Interface
with gr.Blocks(title="MultiScope", theme=theme, css=css) as demo:
    
    # Main Interface
    gr.Markdown("# Multiscope: A Multi-Label Text Classification Dashboard")

    
    with gr.Tab("Pipeline"):
        # Dataset loading
        gr.Markdown("## Data Loading")
        with gr.Row(variant='panel'):
            dataset_source = gr.Radio(["Local", "HuggingFace"], label="Dataset Source", value="Local", info="""Upload your own corpus or use a publicly available dataset from the HuggingFace hub.""")
            dataset_path = gr.Textbox(label="Dataset Path",  info="Enter the path to your local dataset or HuggingFace dataset.")
            hf_subset = gr.Textbox(label="Subset",  info="If applicable, select the subset of the HuggingFace dataset.", visible=False) 
            
            label_col_name = gr.Textbox(label="Label Column",  info="Enter the name of the label column to be used.", visible=False, value='labels') 
            text_col_name = gr.Textbox(label="Text Column",  info="Enter the name of the text column to be used.", visible=False, value='text') 
            
            operations = gr.CheckboxGroup(choices=["Train", "Test", "Make Validation Set", "Make Test Set"], value=["Train", "Test"], label="Data Operations", info="Select the operations to be done.")
            val_portion = gr.Number(label="Validation Set Size", value=0.15, interactive=True, visible=False)
            test_portion = gr.Number(label="Test Set Size", value=0.15, interactive=True, visible=False)
        
        with gr.Row():
            load_data_button = gr.Button("Load Data")

        # Data statistics
        with gr.Accordion(open=True, visible=False) as dataset_statistics:
            gr.Markdown("### Data Statistics")
            with gr.Row("Dataset Rows"):
                train_df = gr.Dataframe(visible=False, interactive=False)
                val_df = gr.Dataframe(visible=False, interactive=False)
                test_df = gr.Dataframe(visible=False, interactive=False)
                display_df = gr.Dataframe(label="Dataset", visible=False, interactive=False)

            with gr.Row("Dataset Stats"):   
                label_stats = gr.Dataframe(label="Label Stats", visible=False, interactive=False)
                token_stats = gr.Dataframe(label="Token Stats", visible=False, interactive=False)
        
            with gr.Row("Graphs"):
                label_counts_plot = gr.Plot(label="Class Counts", visible=False)
                correlation_matrix_plot = gr.Plot(label="Co-occurrence Matrix", visible=False)

        # Classification
        gr.Markdown("## Classification and Inference")
        with gr.Row(variant='panel', equal_height=True):
            clf_method = gr.Radio(["Fine-tune Transformer", "Train SVM"], label="Classification Method", value="Fine-tune Transformer", info='Fine-tune a transformer or train an SVM.')
            output_dir = gr.Textbox(label="Output Directory", interactive=True, info='Enter path to output directory.')
  
        with gr.Row(variant='panel', equal_height=True):
            # transformers
            model_name = gr.Textbox(label="Model name", value='roberta-base', interactive=True)
            batch_size = gr.Textbox(label="Batch size", value=8, interactive=True)
            max_length = gr.Number(label= "Max Sequence Length", minimum=1, value=512, interactive=True)
            n_epochs = gr.Textbox(label="Number of Training Epochs", value=5, interactive=True)
            learning_rate = gr.Textbox(label="Learning rate", value=5e-5, interactive=True)

            # classical ML
            data_language =  gr.Textbox(label="Language of Data", interactive=True, visible=False, info='Enter the language of the training data.') # to-do
            stopwords_path = gr.Textbox(label="Path to Stopwords File", interactive=True, visible=False, info='Enter path to a file containing stopwords.')
            trained_model_path = gr.Textbox(label="Path to Trained Model", value='N/A', interactive=False, visible=False, info='Only applicable if a model has been trained.')
            gridsearch_method = gr.Dropdown(["No Gridsearch", "Standard", "Custom"], label="Gridsearch Method", value="No Gridsearch", visible=False, info='Select a method to optimize hyperparameters.')
        

        with gr.Row(variant='panel', equal_height=True, visible=False) as gridsearch_params:
            with gr.Row(equal_height=True):
                gs_ngram_range = gr.Checkbox(label='N-gram Range')
                n_ngram_range = gr.Number(minimum=1, value=5, interactive=True, show_label=False)

            with gr.Row(equal_height=True):
                gs_min_df = gr.Checkbox(label='Min DF')
                n_min_df = gr.Number(minimum=1, value=5, interactive=True, show_label=False)

            with gr.Row(equal_height=True):
                gs_max_df = gr.Checkbox(label='Max DF')
                n_max_df = gr.Number(minimum=1, value=5, interactive=True, show_label=False)

            with gr.Row(equal_height=True):
                gs_svm_c = gr.Checkbox(label='C (SVM)')
                n_svm_c = gr.Number(minimum=1, value=5, interactive=True, show_label=False)

            with gr.Row(equal_height=True):
                gs_svm_max_iter = gr.Checkbox(label='Max iterations (SVM)')
                n_svm_max_iter = gr.Number(minimum=1, value=5, interactive=True, show_label=False)

        
        # with gr.Row():
        #     generate_button = gr.Button("Generate WandB Report")

        with gr.Row():
            train_model_button = gr.Button("Run", interactive=False)

        with gr.Accordion("Weights & Biases Report:", open=True, visible=False) as report_row:
            wandb_report = gr.HTML()
  
        # displays progress
        loader = gr.Markdown(value="Training model...", visible=False)

        # classification results
        with gr.Accordion(open=True, visible=False) as results_row:
            gr.Markdown("### Classification and Inference")
            with gr.Row(equal_height=True):
                metric_df = gr.Dataframe(label="Results", visible=True)
                report_df = gr.Dataframe(label="Classification Report", visible=True, interactive=False)

            with gr.Row(equal_height=True):
                cnf_matrix = gr.Plot(label="Confusion Matrix", visible=True)
                feature_df = gr.Dataframe(label="Most Informative Features", visible=False, interactive=False) # becomes visible only after training an SVM


# DOCUMENTATION_______________________________________________________________________________________________________________________________

    with gr.Tab("User Guidelines"):
         gr.Markdown("""
        ## General
        Multiscope provides a complete pipeline for multi-label text classification by showing dataset statistics and insights into the label set, in addition to a
        general framework to fine-tune and evaluate state-of-the-art transformer models. The general workflow of this dashboard is the following:
        1. Input the path to a remote or local dataset and load the data.
        2. Select the operations to be performed. Select 'Train' if you want to fine-tune a model on the training data and 'Test' if you want to make predictions with the fine-tuned model on a test set. In the cases where only 'Test' is selected, ensure that you load a valid fine-tuned model!
        3. Input the model that will be used for fine-tuning. Consult the HuggingFace website or the list below for multiple options.
        4. Input the model hyperparameters.
        5. Click 'Train Model' and wait for the model to finish training. Predictions, metrics, classification reports and confusion matrices are saved automatically under the results and visualizations directories.
                     
                     
        ## Dataset Selection
        Multiscope allows for models to be trained on either a local dataset or a dataset available on the HuggingFace hub. Select the dataset source accordingly.
        Local datasets can either be .csv, .xlsx or .json. 
                     
        #### XLSX and CSV files              
        Ensure that the following columns are present in the CSV or Excel files: "text" (contains texts as *strings*), 
        "labels" (contains *lists* of label names) and "split" (can be one of the following strings: "train", "val" or "test"). The test set is not required to contain labels. In this case, Multiscope 
        only performs inference and does not calculate metrics. The validation split ("val") is also not required, since Multi-scope allows for stratified splitting (see below).
                    
        | text | labels                  | split |
        |----- |-------------------------|--------|
        | TEXT | [label 1, label 2, ...] | train |
        | TEXT | [label 1, label 2, ...] | val |
          
        
        #### JSON files
        JSON files should adhere to the following structure:
                     
                {
                    data:{ 
                            'train':    [{'id': ID_1, 'text': TEXT_1, 'labels': [LABELS]}, ..., {'id': ID_N, 'text': TEXT_N, 'labels': [LABELS]} ] 
                            'val':      [{'id': ID_1, 'text': TEXT_1, 'labels': [LABELS]}, ..., {'id': ID_N, 'text': TEXT_N, 'labels': [LABELS]} ]  (if present) 
                            'test':     [{'id': ID_1, 'text': TEXT_1, 'labels': [LABELS]}, ..., {'id': ID_N, 'text': TEXT_N, 'labels': [LABELS]} ]  (remove 'labels' if not present in test set)
                        } 
                }            
        
        ## Data Stratification
        Multiscope also allows you to create a stratified validation split of the training data using the method described in (). For more information about this data
        stratification method, consult the original paper.

        ## Model Selection

        ### A. Fine-tuning a Transformer
        Multiscope is built around the [transformers](https://huggingface.co/docs/transformers/index) library, developed by [HuggingFace](https://huggingface.co/). This means that models that are published on the HuggingFace platform can be used in this dashboard. Below
        are some recommended models for specific use cases. Copy/paste the model names in the *Model Name* text box. Consult the [HuggingFace website](https://huggingface.co/models) for a complete overview of available models. 
                     
        Recommended (English) models:
        * BERT: ```bert-base-cased```, ```bert-large-cased```
        * RoBERTa: ```roberta-base```, ```roberta-large```
        * DistilBERT: ```distilbert/distilbert-base-cased```
        * DeBERTa: ```microsoft/deberta-base```
                     
        Recommended multi-lingual models:
        * XLM-RoBERTa: ```xlm-roberta-base```, ```xlm-roberta-large```
                     
        Recommended language-specific models:
        * BERTje (Dutch): ```GroNLP/bert-base-dutch-cased```
        * CamemBERT (French):```almanach/camembert-base```
        * BERT (German): ```google-bert/bert-base-german-cased```

        Recommended domain-specific models:
        * TwHIN-BERT (Twitter; multilingual): ```Twitter/twhin-bert-base```
        * Sci-BERT (scientific texts): ```allenai/scibert_scivocab_uncased```
        * BioBERT (biomedical texts): ```dmis-lab/biobert-v1.1```
        * FinBERT (financial texts): ```ProsusAI/finbert```
                     
        #### Model Hyperparameters
        * Batch size: Determines the number of texts to be processed in a forward pass and backwards pass. The higher the batch size is, the higher the used GPU memory.
        * Max Sequence Length: Determines the number of tokens to be processed per text. 512 is the maximum for most models, but can be set to a lower number if the texts are short. Ensure to verify this in the Token Stats table. 
            Consult the documentation for each model to verify the maximum sequence length.
        * Epochs: Refers to the number of passes of the entire dataset through the model.
        * Learning Rate: The step size at each iteration.

        ### B. Training an SVM
        Multiscope uses the [scikit-learn](https://scikit-learn.org/stable/) library for building classical machine learning pipelines. In the backend, a Support Vector Machine is trained with a linear kernel using the Binary 
        Relevance framework ([OneVsRest](https://scikit-learn.org/1.5/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)). The data is vectorized using a Tf-idf vectorizer, which assigns higher weights to keywords 
        in each text.
        
        ### Stopwords       
        The user has the option to provide custom stopwords, provided in a .txt file with one word on each new line. 
         If not specified, the stop words default to  the standard list for a language of choice provided by the [NLTK library](https://www.nltk.org/).

        ### Gridsearch
        Gridsearching allows to user to optimize the model's hyperparameters to reach optimal performance. The user can choose between not perfomring a gridsearch, a standard (small-scale) gridsearch,
        or a custom gridsearch where each hyperparameter can be specified in addition to the number of parameters to be gridsearched. Please note that increasing the number of gridsearchable parameters 
        significantly increases training time.

        More specifically, multi-scope performs five-fold cross-validation with stratified splits. For a complete overview of gridsearchable hyperparameters, consult the scikitlearn page for [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), 
        [TfidfVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
                     
        ## Model Evaluation
        The fine-tuned model is evaluated on a validation set during fine-tuning and, if specified, on a held-out (annotated) test set. For this, the standard evaluation metrics
        are employed. These include precision, recall, macro- and micro-averaged F1-score, Exact Match Ratio and Hamming Loss.
                     
        | Metric    | Explanation                                               | 
        |-----------|-----------------------------------------------------------|
        | Precision | The fraction of relevant instances of all retrieved instances |
        | Recall    | The fraction of all relevant instances of all retrieved instances|
        | F1-Score  | The harmonic mean of precision and recall. Macro-averaged F1 is the average of per-class F1-scores. Micro-averaged F1 scores is the globally averaged F1 across instances. |
        | Exact Match Ratio | The fraction of instances where all label sets are predicted correctly. |
        | Hamming Loss | The fraction of incorrectly predicted labels to all correct labels, averaged across instances. |   
        
                     """)
         
    with gr.Tab("About"):
        gr.Markdown("""
        ### Project
        Multiscope is a multi-label text classification dashboard that was developed by [CLiPS](https://www.uantwerpen.be/en/research-groups/clips/) ([University of Antwerp](https://www.uantwerpen.be/en/)) during the [CLARIAH-VL](https://clariahvl.hypotheses.org/) project.
        The code is available here: https://github.com/clips/multiscope.
        
        ### Contact 
        If you have questions, please send them to [Jens Van Nooten](mailto:jens.vannooten@uantwerpen.be) or [Walter Daelemans](mailto:walter.daelemans@uantwerpen.be).

                        """)

    with gr.Row():
        gr.Markdown("""<center><img src="https://platformdh.uantwerpen.be/wp-content/uploads/2019/03/clariah_def.png" alt="Image" width="200"/></center>""")
        gr.Markdown("""<center><img src="https://thomasmore.be/sites/default/files/2022-11/UA-hor-1-nl-rgb.jpg" alt="Image" width="175"/></center>""")


    
# CONTENT VISIBILITY UPDATES___________________________________________________________________________________________________________

    # display errors 
    error_output = gr.Markdown(value="", visible=False)

    # Load data function linking
    load_data_button.click( 
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), 
                    gr.update(visible=True), gr.update(visible=True), 
                    gr.update(visible=False), gr.update(visible=False)),
        inputs=None,
        outputs=[dataset_statistics, display_df, 
                 label_stats, token_stats, 
                 label_counts_plot, correlation_matrix_plot]
    ).then(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=load_data_button
    ).then(
        fn=load_data,
        inputs=[dataset_source, dataset_path, hf_subset, text_col_name, label_col_name, operations, val_portion, test_portion],
        outputs=[train_df, val_df, test_df, display_df,
                 label_stats, token_stats,
                 label_counts_plot, correlation_matrix_plot] 
    ).then(
        fn=lambda: gr.update(interactive=True), # enable train model button after loading data 
        inputs=None, 
        outputs=train_model_button
    ).then(
        fn=lambda: gr.update(interactive=True),
        inputs=None,
        outputs=load_data_button
    )

    # train model or inference
    train_model_button.click(
        fn=lambda: gr.update(interactive=False), # disable button after clicking 
        inputs=None, 
        outputs=train_model_button
    ).then(
        fn=lambda: gr.update(interactive=False), # disable button after clicking 
        inputs=None, 
        outputs=load_data_button
    ).then(
        fn=lambda: gr.update(value="Training model..."),  
        inputs=None, 
        outputs=loader
    ).then(
        fn = wandb_integration, 
        inputs=[dataset_path, clf_method, gridsearch_method, model_name, batch_size, max_length, learning_rate], 
        outputs=[wandb_report]
    ).then(
        fn= toggle_classification_results,
        inputs=[operations],
        outputs=[report_row, results_row, loader] 
    ).then(
        fn= toggle_feature_df,
        inputs=[clf_method, operations],
        outputs=[feature_df] 
    ).then(
        fn=train_model,
        inputs=[output_dir, clf_method, model_name, train_df, val_df, test_df, batch_size, max_length, learning_rate, n_epochs, 
                operations, 
                gridsearch_method, trained_model_path, stopwords_path, data_language,
                gs_ngram_range, n_ngram_range, gs_min_df, n_min_df, gs_max_df, n_max_df, gs_svm_c, n_svm_c, gs_svm_max_iter, n_svm_max_iter],
        outputs=[metric_df, report_df, cnf_matrix, feature_df, error_output]
    ).then(
        fn=show_error,
        inputs=error_output,  # Pass the error message to this function and raise error
    ).then(
        fn=lambda: gr.update(interactive=True),  # enable button after model is done training
        inputs=None, 
        outputs=train_model_button
    ).then(
        fn=lambda: gr.update(interactive=True),  # enable button after model is done training
        inputs=None, 
        outputs=load_data_button
    ).then(
        fn=lambda: gr.update(value="Finished training!"), # update loader 
        inputs=None, 
        outputs=loader
    )

    # change values of items based on selected operations
    dataset_source.change(
        fn=toggle_subset_display,
        inputs=dataset_source,
        outputs=[hf_subset, text_col_name, label_col_name]
    )

    operations.change(
        fn=update_button_text,
        inputs=operations,
        outputs=train_model_button
    )

    operations.change(
        fn=toggle_set_sizes,
        inputs=operations,
        outputs=[val_portion, test_portion]
    )

    operations.change(
        fn=toggle_hyperparameters,
        inputs=operations,
        outputs=[max_length, n_epochs, learning_rate, gridsearch_method, trained_model_path]
    )

    operations.change(
        fn=toggle_data_display,
        inputs=operations,
        outputs=[train_df, label_stats, label_counts_plot, correlation_matrix_plot]
    )

    # change visibility of hyperparameters based on clf method
    clf_method.change(
        fn=toggle_parameter_visibility,
        inputs=[clf_method],
        outputs=[model_name, batch_size, max_length, n_epochs, learning_rate, gridsearch_method, trained_model_path, stopwords_path, data_language, gridsearch_params] 
    )

    gridsearch_method.change(
        fn=toggle_gridsearch_params,
        inputs=[gridsearch_method],
        outputs=[gridsearch_params]
    )


# Launch Gradio app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a Gradio app.")
    parser.add_argument(
        "--public",
        action="store_true",
        help="If set, creates a public URL for the app.",
    )
    args = parser.parse_args()

    # Launch the app with or without a public URL based on the argument
    demo.launch(share=args.public)
