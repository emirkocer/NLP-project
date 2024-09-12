import streamlit as st
import mmlu_inference
import math_inference

"""
Streamlit UI for running evaluations using the open-source
Mistral 7B LLM. Dataset options: MMLU and MATH
"""
def main():
    st.title('Emir Kocer - Codeway NLP Project')
    dataset = st.sidebar.selectbox('Choose a dataset:', ['MMLU', 'MATH'])

    if dataset == 'MMLU':
        st.header('MMLU Dataset')
        evaluation_type = st.radio("Select Evaluation Type:", ['Baseline Evaluation', 'Few-shot Evaluation', 'Few-shot & CoT Evaluation'])
        if st.button('Run Evaluation'):
            accuracy = run_evaluation(dataset, evaluation_type)
            st.write(f'The accuracy of {evaluation_type} on {dataset} dataset is: {accuracy}%')

    elif dataset == 'MATH':
        st.header('MATH Dataset')
        evaluation_type = st.radio("Select Evaluation Type:", ['Baseline Evaluation', 'Few-shot Evaluation' ])
        if st.button('Run Evaluation'):
            accuracy = run_evaluation(dataset, evaluation_type)
            st.write(f'The accuracy of {evaluation_type} on {dataset} dataset is: {accuracy}%')

def run_evaluation(dataset, evaluation_type):
    if dataset == 'MMLU':
        if evaluation_type == 'Baseline Evaluation':
            return mmlu_inference.evaluate("baseline")
        elif evaluation_type == 'Few-shot Evaluation':
            return mmlu_inference.evaluate("few-shot")
        elif evaluation_type == 'Few-shot & CoT Evaluation':
            return mmlu_inference.evaluate("few-shot-and-cot")
    elif dataset == 'MATH':
        if evaluation_type == 'Baseline Evaluation':
            return math_inference.evaluate("baseline")
        elif evaluation_type == 'Few-shot Evaluation':
            return math_inference.evaluate("few-shot")

if __name__ == "__main__":
    main()
