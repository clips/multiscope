# Multiscope

This repository contains the code for a multi-label text classification pipeline. The documentation can be found [here](https://www.uantwerpen.be/en/research-groups/clips/research/computational-linguistics/compling-resources/clips-technical-repo/#:~:text=11.%20Multiscope%3A%20A%20User%2DFriendly%20Multi%2DLabel%20Text%20Classification%20Dashboard).  

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

### How to cite
Jens Van Nooten and Walter Daelemans. 2024. Multiscope: A User-Friendly Multi-Label Text Classification Dashboard. CLiPS Technical Report Series 11 (CTRS 11). ISSN 2033-3544. Computational Linguistics, Psycholinguistics, and Sociolinguistics Research Center.
