import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from utils import (load_data, train_model)
from gradio_utils import (wandb_report, toggle_parameter_visibility,
                   show_error, update_button_text, toggle_hyperparameters, toggle_data_display, toggle_classification_results, toggle_subset_display)
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
}
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
            dataset_source = gr.Radio(["Local", "HuggingFace"], label="Dataset Source:", value="Local", info="""Upload your own corpus or use a publicly available dataset from the HuggingFace hub.""")
            dataset_path = gr.Textbox(label="Dataset Path:",  info="Enter the path to your local dataset or HuggingFace dataset.")
            hf_subset = gr.Textbox(label="Subset:",  info="If applicable, select the subset of the HuggingFace dataset.", visible=False) 
            operations = gr.CheckboxGroup(choices=["Train", "Test", "Split Training Data"], value=["Train", "Test"], label="Data Operations", info="Select the operations to be done.")
        
        with gr.Row():
            load_data_button = gr.Button("Load Data")

        # Data statistics
        with gr.Accordion("Data Statistics", open=True, visible=False) as dataset_statistics:
            with gr.Row("Dataset Rows"):
                train_df = gr.Dataframe(label="Training Dataset", visible=False)
                val_df = gr.Dataframe(label="Validation Dataset", visible=False)
                test_df = gr.Dataframe(label="Test Dataset", visible=False)

            with gr.Row("Dataset Stats"):   
                label_stats = gr.Dataframe(label="Label Stats", visible=False)
                token_stats = gr.Dataframe(label="Token Stats", visible=False)
        
            with gr.Row("Graphs"):
                label_counts_plot = gr.Plot(label="Class Counts", visible=False)
                correlation_matrix_plot = gr.Plot(label="Co-occurrence Matrix", visible=False)

        # Classification
        gr.Markdown("## Classification and Inference")
        with gr.Row(variant='panel'):
            clf_method = gr.Radio(["Fine-tune", "Prompt LLM"], label="Select Classification Method", value="Fine-tune")
            model_name = gr.Textbox(label="Model name:", value='roberta-base', interactive=True)
            batch_size = gr.Textbox(label="Batch size:", value=8, interactive=True)
            n_epochs = gr.Textbox(label="Number of Training Epochs:", value=5, interactive=True)
            learning_rate = gr.Textbox(label="Learning rate:", value=5e-5, interactive=True)
            # to-do: implement early stopping 

        with gr.Row():
            train_model_button = gr.Button("Run", interactive=False)

        # WandB integration
        demo.integrate(wandb=wandb)

        wandb.init(
            project="multiscope-demo",
            name='test',
            tags=["baseline"],
            group="bert",
        )
        
        url = wandb.run.get_url()

        with gr.Accordion("Weights & Biases Report:", open=True, visible=False) as report_row:
            iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
            report = gr.HTML(iframe, visible=True)

        # displays progress
        loader = gr.Markdown(value="Training model...", visible=False)


        # classification results
        with gr.Accordion("Classification Results", open=True, visible=False) as results_row:
            with gr.Row():
                metric_df = gr.Dataframe(label="Results", visible=True)
                report_df = gr.Dataframe(label="Classification Report", visible=True)

            with gr.Row():
                cnf_matrix = gr.Plot(label="Confusion Matrix", visible=True)


# DOCUMENTATION_______________________________________________________________________________________________________________________________

    with gr.Tab("User Guidelines"):
         gr.Markdown("""
        ### General
        Multiscope provides a complete pipeline for multi-label text classification by showing dataset statistics and insights into the label set, in addition to a
        general framework to fine-tune and evaluate state-of-the-art transformer models. The general workflow of this dashboard is the following:
        1. Input the path to a remote or local dataset and load the data.
        2. Select the operations to be performed. Select 'Train' if you want to fine-tune a model on the training data and 'Test' if you want to make predictions with the fine-tuned model on a test set. In the cases where only 'Test' is selected, ensure that you load a valid fine-tuned model!
        3. Input the model that will be used for fine-tuning. Consult the HuggingFace website or the list below for multiple options.
        4. Input the model hyperparameters.
        5. Click 'Train Model' and wait for the model to finish training. Predictions, metrics, classification reports and confusion matrices are saved automatically under the results and visualizations directories.
                     
                     
        ### Dataset Selection
        Multiscope allows for models to be trained on either a local dataset or a dataset available on the HuggingFace hub. Select the dataset source accordingly.
        Local datasets can either be .csv, .xlsx or .json. 
                     
        ##### XLSX and CSV files              
        Ensure that the following columns are present in the CSV or Excel files: "text" (contains texts as strings), 
        "labels" (contains *lists* of label names) and "split" (can either be "train", "val" or "test"). The test set is not required to contain labels. In this case, Multiscope 
        only performs inference and does not calculate metrics.
                    
        | text | labels                  | split |
        |----- |-------------------------|--------|
        | TEXT | [label 1, label 2, ...] | train |
        | TEXT | [label 1, label 2, ...] | val |
          
        
        ##### JSON files
        JSON files should adhere to the following structure:
                     
                {
                    data:{ 
                            'train':    [{'id': ID_1, 'text': TEXT_1, 'labels': [LABELS]}, ..., {'id': ID_N, 'text': TEXT_N, 'labels': [LABELS]} ] 
                            'val':      [{'id': ID_1, 'text': TEXT_1, 'labels': [LABELS]}, ..., {'id': ID_N, 'text': TEXT_N, 'labels': [LABELS]} ]  (if present) 
                            'test':     [{'id': ID_1, 'text': TEXT_1, 'labels': [LABELS]}, ..., {'id': ID_N, 'text': TEXT_N, 'labels': [LABELS]} ]  (remove 'labels' if not present in test set)
                        } 
                }            
        
        ### Data Stratification
        Multiscope also allows you to create a stratified validation split of the training data using the method described in (). For more information about this data
        stratification method, consult the original paper.

        ### Model Selection
        Multiscope is built around the *transformers* library, developed by HuggingFace. This means that models that are published on the HuggingFace platform can be used in this dashboard. Below
        are some recommended models for specific use cases. Copy/paste the model names in the *Model Name* text box. 
                     
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
                     
        ### Model Evaluation
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
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=None,

        outputs=[dataset_statistics, train_df, val_df, test_df, 
                 label_stats, token_stats, label_counts_plot, correlation_matrix_plot]
    ).then(
        fn=load_data,
        inputs=[dataset_source, dataset_path, hf_subset, operations],
        outputs=[train_df, val_df, test_df, label_stats, 
                 token_stats, label_counts_plot, correlation_matrix_plot] 
    ).then(
        fn=lambda: gr.update(interactive=True), # enable train model button after loading data 
        inputs=None, 
        outputs=train_model_button
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
        fn= toggle_classification_results,
        inputs=operations,
        outputs=[report_row, results_row, loader] 
    ).then(
        fn=train_model,
        inputs=[clf_method, model_name, train_df, val_df, test_df, batch_size, learning_rate, n_epochs, operations],
        outputs=[metric_df, report_df, cnf_matrix, error_output]
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
        outputs=[hf_subset]
    )

    operations.change(
        fn=update_button_text,
        inputs=operations,
        outputs=train_model_button
    )

    operations.change(
        fn=toggle_hyperparameters,
        inputs=operations,
        outputs=[n_epochs, learning_rate]
    )

    operations.change(
        fn=toggle_data_display,
        inputs=operations,
        outputs=[train_df, label_stats, label_counts_plot, correlation_matrix_plot]
    )

    # change visibility of hyperparameters based on clf method
    clf_method.change(
        toggle_parameter_visibility,
        inputs=[clf_method],
        outputs=[batch_size, n_epochs, learning_rate] 
    )


# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
