import gradio as gr
import pandas as pd
from datasets import load_dataset
import plotly.graph_objects as go
from utils import create_cooccurrence_matrix, load_data, train_model, wandb_report, toggle_parameter_visibility, load_huggingface_dataset, load_local_dataset
from finetune import finetune_transformer
import torch
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

def show_error(msg):
    if msg:
        print('triggered')
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


#APP INTERFACE______________________________________________________________________________________________

# Gradio Interface
with gr.Blocks(title="MultiScope", theme=theme, css=css) as demo:
    
    gr.Markdown("# Multi-Scope: A Multi-Label Text Classification Dashboard")

    # Dataset loading
    with gr.Row(variant='panel'):
        dataset_source = gr.Radio(["Local", "HuggingFace"], label="Dataset Source:", value="Local")
        dataset_path = gr.Textbox(label="Dataset Path:")
        operations = gr.CheckboxGroup(choices=["Train", "Test", "Split Training Data"], 
                                 value=["Train", "Test"], 
                                 label="Data Operations")
    
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
    with gr.Row(variant='panel'):
        clf_method = gr.Radio(["Fine-tune", "Prompt LLM"], label="Select Classification Method", value="Fine-tune")
        model_name = gr.Textbox(label="Model name:", value='roberta-base', interactive=True)
        batch_size = gr.Textbox(label="Batch size:", value=8, interactive=True)
        n_epochs = gr.Textbox(label="Number of Training Epochs:", value=5, interactive=True)
        learning_rate = gr.Textbox(label="Learning rate:", value=5e-5, interactive=True)
        # to-do: implement early stopping 
        #to-do: custom losses

    # change visibility of hyperparameters based on clf method
    clf_method.change(
        toggle_parameter_visibility,
        inputs=[clf_method],
        outputs=[batch_size, n_epochs, learning_rate] 
    )

    with gr.Row():
        train_model_button = gr.Button("Run")

    # Load data function linking
    load_data_button.click( 
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
        inputs=None,
        outputs=[dataset_statistics, train_df, val_df, test_df, 
                 label_stats, token_stats, label_counts_plot, correlation_matrix_plot]
    ).then(
        fn=load_data,
        inputs=[dataset_source, dataset_path, operations],
        outputs=[train_df, val_df, test_df, label_stats, 
                 token_stats, label_counts_plot, correlation_matrix_plot] 
    )

    # WandB integration
    demo.integrate(wandb=wandb)
    wandb.init(
        project="multiscope-demo",
        name='test',
        tags=["baseline"],
        group="bert",
    )
    
    # wandb integration
    url = wandb.run.get_url()
    with gr.Accordion("Weights & Biases Report:", open=True, visible=False) as report_row:
        iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
        report = gr.HTML(iframe, visible=True)

    loader = gr.Markdown(value="Training model...", visible=False)

    with gr.Accordion("Classification Results", open=True, visible=False) as results_row:
        with gr.Row():
            metric_df = gr.Dataframe(label="Results", visible=True)
            report_df = gr.Dataframe(label="Classification Report", visible=True)

        with gr.Row():
            cnf_matrix = gr.Plot(label="Confusion Matrix", visible=True)

    # display errors 
    error_output = gr.Markdown(value="", visible=False)

    train_model_button.click(
        fn=lambda: gr.update(interactive=False), # disable button after clicking 
        inputs=None, 
        outputs=train_model_button
    # ).then(
    #     fn= lambda: (gr.update(visible=True), gr.update(visible=True)),
    #     inputs=None,
    #     outputs=[report_row, results_row] 
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
        inputs=error_output,  # Pass the error message to this function
    ).then(
        fn=lambda: gr.update(interactive=True),  # enable button after model is done training
        inputs=None, 
        outputs=train_model_button
    ).then(
        fn=lambda: gr.update(value="Finished training!"),  # enable button after model is done training
        inputs=None, 
        outputs=loader
    )

    # change values of items based on selected operations
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



# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
