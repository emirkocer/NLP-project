from huggingface_hub import InferenceClient
from prompts import math_baseline_prompt_template, math_verifier_template
import argparse
from openai import OpenAI
import random
import json
from prompts import build_prompt
import prompts
import os
import utils_inference
import numpy as np
from dotenv import load_dotenv
import streamlit as st

MAX_TOKEN=512 

""" Load JSON Lines file and return a list of dictionaries. """
def load_jsonl_data(file_path):    
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

""" Returns the MATH baseline prompt """
def get_baseline_prompt(problem):
    prompt = build_prompt(
            prompt=prompts.math_baseline_prompt_template,
            input_vars={"problem": problem}
            )
    return prompt

""" Returns the MATH few-shot prompt """
def get_few_shot_prompt(few_shot_examples, problem):
    prompt = ""
    prompt += prompts.math_few_shot_prompt_intro
    few_shot_examples_list = list(few_shot_examples)
    # Add few-shot examples to the prompt
    for example in few_shot_examples_list:
        prompt += f"Problem: {example['problem']}\nSolution: {example['solution']}\nAnswer: {example['answer']}\n"
    # Add the new question
    prompt += build_prompt(
            prompt=prompts.math_few_shot_prompt_template,
            input_vars={"problem": problem}
            )
    return prompt


""" Evaluates Mistral 7B on the MMLU dataset based on the desired mode 
and returns the accuracy
"""
def evaluate(mode):

    print(f"Running in {mode} mode")

    load_dotenv() # load API keys
    hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_client = OpenAI() # OpenAI client for GPT calls
   
    test_data_path = 'math_splits/test.jsonl' # MATH test data from PRM800K
    math_test = load_jsonl_data(test_data_path)

    # five random examples from MATH train data, for few-shotting
    train_data_path = 'math_splits/train.jsonl' 
    math_train = load_jsonl_data(train_data_path)  
    few_shot_examples = random.sample(math_train, 5)

    correct_answers = 0
    total_questions = len(math_test) # total number of test questions
    processed_questions = 0

    for i in range(total_questions):
        if mode == 'baseline':
            '''
            Part 1: Baseline accuracy
            '''
            prompt = get_baseline_prompt(math_test[i]['problem'])
            model = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=hf_api_key ) # HF Inference API
        elif mode == 'few-shot':
            '''
            Part 2: Few-shot accuracy
            '''
            prompt = get_few_shot_prompt(few_shot_examples, math_test[i]['problem'])
            model = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=hf_api_key )
      
        model_answer = utils_inference.get_model_response(model, prompt, MAX_TOKEN)
        gpt_response = utils_inference.call_gpt4_verifier(openai_client, 
                                                          prompts.math_verifier_template,
                                                          model_answer, 
                                                          math_test[i]['answer']
                                                          )
        # Check if the predicted answer is correct
        if gpt_response == 'correct':
            correct_answers += 1
        processed_questions += 1
        percentage_processed = (processed_questions/total_questions) * 100
        st.write(f'Evaluated percentage : {percentage_processed}')
        if percentage_processed == 100.0:
            st.write(f'Evaluation is completed !')
        print(correct_answers / (i + 1))
    accuracy = correct_answers / total_questions
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MMLU model inference")
    parser.add_argument('--mode', type=str, choices=['baseline', 'few-shot'],
                        help='Select the mode to run the inference on', required=True)
    
    args = parser.parse_args()
    accuracy = evaluate(mode=args.mode)
    print(f"Accuracy for {args.mode} mode is: {accuracy}")
    accuracy = evaluate(mode="baseline")
    print(accuracy)





