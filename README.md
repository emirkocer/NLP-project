# NLP Test Project

## Overview
This project is designed to evaluate the open-source Mistral 7B model on two datasets, 
MMLU and MATH, using various evaluation techniques. 
Users can run evaluations either through a Streamlit UI for an easy and interactive experience 
or by running scripts directly from the terminal for more controlled testing.

Note: In this project, OpenAI and HuggingFace APIs are used for text generation.
WandB API is used for output generation.
Please set the .env file using your API keys.
All API key variables are left empty.

## Installation
Clone the repository and install the required packages:

git clone git@github.com:emirkocer/NLP-test-project.git project
cd project   
pip install -r requirements.txt   

## Running the Streamlit App
To start the Streamlit UI, navigate to the project directory and run:

streamlit run app.py

On the UI, you can select a dataset and an evaluation mode to run the desired inference.

## Running Inferences from Terminal
For direct command line interactions, you can use the provided Bash scripts to run evaluations:

chmod +x run_evaluation.sh   
./run_evaluation.sh mmlu <evaluation_mode>   
./run_evaluation.sh math <evaluation_mode>   

Available evaluation modes for MMLU: 'baseline', 'few-shot' and 'few-shot-and-cot'
Available evaluation modes for MATH: 'baseline' and 'few-shot' 

## Fine-Tuning Models

In the finetune directory, you will find subfolders for both MATH and MMLU datasets containing 
the necessary Python files for fine-tuning Mistral 7B base model on a GPU using the UnSloth library
for training optimization.

Two Colab notebooks are provided for inference with both fine-tuned models.


