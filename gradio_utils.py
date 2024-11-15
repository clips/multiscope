import gradio as gr

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

def toggle_subset_display(dataset_source):
    if dataset_source == 'HuggingFace':
         return gr.update(visible=True)
    else:
        return gr.update(visible=False)
               
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