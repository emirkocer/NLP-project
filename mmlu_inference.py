from huggingface_hub import InferenceClient
from datasets import load_dataset, concatenate_datasets, DatasetDict
from openai import OpenAI
import utils_inference
import os
from prompts import build_prompt, mmlu_verifier_template
import prompts
import argparse
from dotenv import load_dotenv
import streamlit as st

MAX_TOKEN=100

""" Returns the MMLU baseline prompt """
def get_baseline_prompt(question, options):
    formatted_options = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
    prompt = build_prompt(
            prompt=prompts.mmlu_baseline_prompt_template,
            input_vars={"question": question, "formatted_options": formatted_options}
            )
    return prompt

""" Returns the MMLU five-shot prompt """
def get_few_shot_prompt(few_shot_examples, intro_prompt, question, options):
    options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
    prompt = ""
    prompt += intro_prompt
    few_shot_examples_list = list(few_shot_examples)
    # Add few-shot examples to the prompt
    for example in few_shot_examples_list:
        options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(example['choices'])])
        prompt += f"Question: {example['question']}\n{options_text}\nAnswer: {chr(65 + int(example['answer']))}\n\n"
    # Add the new question
    formatted_options = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
    prompt += build_prompt(
            prompt=prompts.mmlu_few_shot_prompt_template,
            input_vars={"question": question, "formatted_options": formatted_options}
            )
    return prompt

""" Returns the MMLU Few-shot + CoT prompt """
def get_cot_prompt(question, options):
    formatted_options = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
    prompt = build_prompt(
            prompt=prompts.mmlu_cot_few_shot_prompt_template,
            input_vars={"question": question, "formatted_options": formatted_options}
            )
    return prompt

""" Loads MMLU dataset """
def load_mmlu_dataset():
    ds_chemistry = load_dataset("cais/mmlu", "high_school_chemistry")
    ds_biology = load_dataset("cais/mmlu", "high_school_biology")
    ds_physics = load_dataset("cais/mmlu", "high_school_physics")

    # Combine them into a single DatasetDict
    combined_test = concatenate_datasets([ds_chemistry['test'], ds_biology['test'], ds_physics['test']])
    combined_validation = concatenate_datasets([ds_chemistry['validation'], ds_biology['validation'], ds_physics['validation']])
    combined_dev = concatenate_datasets([ds_chemistry['dev'], ds_biology['dev'], ds_physics['dev']])

    dataset = DatasetDict({
        'test': combined_test,
        'validation': combined_validation,
        'dev': combined_dev
    })
    return dataset

""" Loads MMLU few-shot examples """
def load_mmlu_fs_dataset():
    ds_chemistry = load_dataset("cais/mmlu", "high_school_chemistry")
    ds_biology = load_dataset("cais/mmlu", "high_school_biology")
    ds_physics = load_dataset("cais/mmlu", "high_school_physics")

    fs_chemistry = ds_chemistry["dev"]
    fs_biology = ds_biology["dev"]
    fs_physics = ds_physics["dev"]
    
    return fs_chemistry, fs_biology, fs_physics

""" Evaluates Mistral 7B on the MMLU dataset based on the desired mode 
and returns the accuracy
"""
def evaluate(mode):

    print(f"Running in {mode} mode")

    load_dotenv() # load API keys
    hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_client = OpenAI() # for evaluation
    dataset = load_mmlu_dataset()
    fs_chemistry, fs_biology, fs_physics = load_mmlu_fs_dataset() # load few-shot data once here
    test_data = dataset["test"]
    processed_questions = 0
    total_questions = len(test_data['question'])
    correct_answers = 0
    for i in range(total_questions):
        if mode == 'baseline':
            '''
            Part 1: Baseline 
            '''
            model = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=hf_api_key)
            prompt = get_baseline_prompt(test_data['question'][i], test_data['choices'][i])
        elif mode == 'few-shot':
            '''
            Task 2: Prompting Techniques
            '''
            subject = test_data['subject'][i]
            if subject == 'high_school_chemistry':
                fs_data = fs_chemistry
                subject = 'chemistry'
            elif subject == 'high_school_biology':
                fs_data = fs_biology
                subject = 'biology'
            elif subject == 'high_school_physics':
                fs_data = fs_physics
                subject = 'physics'
            few_shot_examples = fs_data
            model = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=hf_api_key)
            intro_prompt = prompts.mmlu_few_shot_prompt_intro.format(subject=subject)
            prompt = get_few_shot_prompt(few_shot_examples, intro_prompt, test_data['question'][i], test_data['choices'][i])
        elif mode == 'few-shot-and-cot': # experimental
            model = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=hf_api_key)
            prompt = get_cot_prompt(test_data['question'][i], test_data['choices'][i])

        model_answer = utils_inference.get_model_response(model, prompt, MAX_TOKEN)
        gpt_response = utils_inference.call_gpt4_verifier(openai_client, 
                                                          prompts.mmlu_verifier_template,
                                                          model_answer, 
                                                          test_data['answer'][i]
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
    parser.add_argument('--mode', type=str, choices=['baseline', 'few-shot', 'few-shot-and-cot'],
                        help='Select the mode to run the inference on', required=True)
    
    args = parser.parse_args()
    accuracy = evaluate(mode=args.mode)
    print(f"Accuracy for {args.mode} mode is: {accuracy}")
    


