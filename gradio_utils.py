import gradio as gr
import wandb
from utils import generate_run_name

def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe, visible=False)


def wandb_integration(dataset_path, clf_method, gridsearch_method, model_name, batch_size, max_length, learning_rate):
    dataset_name = dataset_path.split('/')[-1].split('.')[0] if '/' in dataset_path else dataset_path.split('.')[0]
    run_name = generate_run_name(clf_method, gridsearch_method, model_name, batch_size, max_length, learning_rate)
    
    run = wandb.init(
        project=f"multiscope-{dataset_name}",
        name=run_name,  # Use the generated run name
        reinit=True  # To allow multiple runs in a single session
    )
    
    url = wandb.run.get_url()

    iframe = f'<iframe src="{url}" style="border:none;height:1024px;width:100%"></iframe>'
    
    return gr.update(value=iframe, visible=True)


def toggle_parameter_visibility(clf_method):
    if clf_method == 'Fine-tune Transformer':
        return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(value='No Gridsearch', visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    else:
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value='No Gridsearch', visible=True),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
    
def show_error(msg):
    if msg:
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

def toggle_subset_display(dataset_source):
    if dataset_source == 'HuggingFace':
         return gr.update(visible=True)
    else:
        return gr.update(visible=False)
               
def toggle_hyperparameters(operations):
    if "Test" in operations and "Train" not in operations:
        return gr.update(interactive=True), gr.update(value='N/A', interactive=False), gr.update(value='N/A', interactive=False), gr.update(interactive=False), gr.update(value='', interactive=True),
    elif "Train" in operations:
        return gr.update(interactive=True), gr.update(value=5, interactive=True), gr.update(value=5e-5, interactive=True), gr.update(interactive=True), gr.update(value='N/A', interactive=False)
    else:
        return gr.update(interactive=True), gr.update(value=5, interactive=True), gr.update(value=5e-5, interactive=True), gr.update(interactive=True), gr.update(value='N/A', interactive=False)
    
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
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    

def toggle_gridsearch_params(gridsearch_method):
    if gridsearch_method == 'Custom':
        return(gr.update(visible=True))
    else:
        return(gr.update(visible=False))
    
def toggle_feature_df(clf_method, operations):
    if clf_method == 'Train SVM' and 'Train' in operations:
        return(gr.update(visible=True))
    else:
        return(gr.update(visible=False))
    
    
