# Multiscope

This repository contains the code for a multi-label text classification pipeline.

### Installation
1. Create a new conda environment: ```conda create -n multiscope python=3.12.5```
2. Activate the environment: ```conda activate multiscope```
3. Clone the repository: ```git clone https://github.com/clips/multiscope.git```
4. Change the working directory: ```cd multiscope```
5. Install the requirements: ```pip3 install -r requirements.txt```

### Google Colab
Use Multiscope in Google Colab: https://colab.research.google.com/drive/1NexZF1CtWdsxkLn0hqFT7Y12T9um4oee?usp=sharing

### User interface
1. Activate the ```multiscope``` conda environment: ```conda activate multiscope```
2. First, run ```python setup_nltk.py``` to install all necessary NLTK-related files.
3. To run the pipeline in a Gradio User Interface, run (CUDA_VISIBLE_DEVICES=X) ```python app.py``` and browse to http://127.0.0.1:7860. 


