import streamlit as st
import pandas as pd
from datasets import load_dataset
import plotly.graph_objects as go
from graph_utils import create_cooccurrence_matrix
from utils import load_huggingface_dataset, load_local_dataset
from finetune import train_and_eval
import torch

pd.options.plotting.backend = "plotly"

@st.cache_resource
def fine_tune_transformer(train_df, val_df, test_df, model_name, use_custom_loss=False, **kwargs):
    train_and_eval(train_df, val_df, test_df, model_name, **kwargs)

################################################################################################################################

# States

if 'loading_data_complete' not in st.session_state:
    st.session_state['loading_data_complete'] = False

if 'train_df' not in st.session_state:
    st.session_state['train_df'] = pd.DataFrame()

if 'val_df' not in st.session_state:
    st.session_state['val_df'] = pd.DataFrame()

if 'test_df' not in st.session_state:
    st.session_state['test_df'] = pd.DataFrame()

if 'label_counts' not in st.session_state:
    st.session_state['label_counts'] = None

if 'correlation_matrix' not in st.session_state:
    st.session_state['correlation_matrix'] = None

################################################################################################################################

st.set_page_config(layout="wide")

st.title("Multi-Task: A Multi-Label Text Classification Dashboard")

# LOADING THE DATA

st.header("Loading the Dataset")

dataset_source = st.radio(
    "Select the source of the dataset:",
    key="data_selection",
    options=["HuggingFace", "Local file"],
    captions=['Select a dataset available on HuggingFace.', 'Loads a local dataset.']
)

dataset_path = st.text_input('Type the path to the dataset:')

if st.button('Load data', key="load_data"):
    with st.status("Processing data..."):
        st.write("Loading dataset...")

        if dataset_source == "HuggingFace":
            train_df = load_huggingface_dataset(dataset_path, subset='train')
            val_df = load_huggingface_dataset(dataset_path, subset='val')
            test_df = load_huggingface_dataset(dataset_path, subset='test')

        else:
            train_df = load_local_dataset(dataset_path, subset='train')
            val_df = load_local_dataset(dataset_path, subset='val')
            test_df = load_local_dataset(dataset_path, subset='test')

        if not train_df.empty:
            st.write("Generating graphs...")
            label_counts = train_df['labels'].explode().value_counts().plot.bar(title='Class counts')
            label_counts.update_layout(xaxis_type='category')

            correlation_matrix = create_cooccurrence_matrix(train_df.labels)
            st.write("Done!")

            # Store in session state
            st.session_state['train_df'] = train_df
            st.session_state['val_df'] = val_df
            st.session_state['test_df'] = test_df
            st.session_state['label_counts'] = label_counts
            st.session_state['correlation_matrix'] = correlation_matrix
            st.session_state['loading_data_complete'] = True

        else:
            st.write('ERROR: Invalid data path.')

# Use session state variables
if st.session_state['loading_data_complete'] and not st.session_state['train_df'].empty:
    st.write("Dataset:")
    st.write(st.session_state['train_df'].head())

    container = st.container()

    with container:
        lcol, rcol = st.columns(2)

    with lcol:
        st.plotly_chart(st.session_state['label_counts'])

    with rcol:
        st.plotly_chart(st.session_state['correlation_matrix'])

# TRAINING A MODEL

clf_method = st.radio(
    "Select the classification method:",
    key="clf_method",
    options=["Fine-tune", "Prompt LLM", "Distance-based Classification"],
)

model_name = st.text_input('Type the path the model name:')

print(torch.cuda.is_available())
   
if st.button('Train model', key='train_model'):
    if clf_method == 'Fine-tune':
        with st.status(f"Fine-tuning {model_name}..."):
            fine_tune_transformer(st.session_state['train_df'], st.session_state['val_df'], st.session_state['test_df'], model_name, use_custom_loss=False)

        # customize training args
        # customize output directory name

    if clf_method == 'Prompt LLM':
        with st.status(f"Classifying data with {model_name}..."):
            pass

    if clf_method == 'Distance-based Classification':
        with st.status(f"Classifying data with {model_name}..."):
            pass




