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
    print('triggered')
    raise(gr.Error(msg))


#APP INTERFACE______________________________________________________________________________________________

# Gradio Interface
with gr.Blocks(title="MultiScope", theme=theme, css=css) as demo:
    
    gr.Markdown("# Multi-Scope: A Multi-Label Text Classification Dashboard")

    # Dataset loading
    with gr.Row(variant='panel'):
        dataset_source = gr.Radio(["Local", "HuggingFace"], label="Dataset Source:", value="Local")
        dataset_path = gr.Textbox(label="Dataset Path:")

    with gr.Row():
        load_data_button = gr.Button("Load Data")

    # Data statistics
    with gr.Accordion("Training Data Statistics", open=True, visible=False) as dataset_statistics:
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
        batch_size = gr.Number(label="Batch size:", value=8, interactive=True)
        n_epochs = gr.Number(label="Number of Training Epochs:", value=5, interactive=True)
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
        train_model_button = gr.Button("Train Model")


    # Load data function linking
    load_data_button.click( 
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
        inputs=None,
        outputs=[dataset_statistics, train_df, val_df, test_df, 
                 label_stats, token_stats, label_counts_plot, correlation_matrix_plot]
    ).then(
        fn=load_data,
        inputs=[dataset_source, dataset_path],
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

    url = wandb.run.get_url()
    with gr.Accordion("Weights & Biases Report:", open=True, visible=False) as report_row:
        iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
        report = gr.HTML(iframe, visible=True)

    with gr.Accordion("Classification Results", open=True, visible=False) as results_row:
        with gr.Row():
            metric_df = gr.Dataframe(label="Results", visible=True)
            report_df = gr.Dataframe(label="Classification Report", visible=True)

        with gr.Row():
            cnf_matrix = gr.Plot(label="Confusion Matrix", visible=True)

    error_output = gr.Markdown(value="", visible=False)
    result_output = gr.Textbox(label="Result")

    train_model_button.click(
        fn=lambda: gr.update(interactive=False), # disable button after clicking 
        inputs=None, 
        outputs=train_model_button
    ).then(
        fn= lambda: (gr.update(visible=True), gr.update(visible=True)),
        inputs=None,
        outputs=[report_row, results_row] 
    ).then(
        fn=train_model,
        inputs=[clf_method, model_name, train_df, val_df, test_df, batch_size, learning_rate],
        outputs=[metric_df, report_df, cnf_matrix, error_output]
    ).then(
        fn=show_error,
        inputs=error_output,  # Pass the error message to this function
    ).then(
        fn=lambda: gr.update(interactive=True),  # enable button after model is done training
        inputs=None, 
        outputs=train_model_button
    )

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
