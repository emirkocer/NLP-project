#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <evaluation_mode>"
    exit 1
fi

dataset=$1
evaluation_mode=$2

if [ "$dataset" = "mmlu" ]; then
    python3 mmlu_inference.py --mode $evaluation_mode
elif [ "$dataset" = "math" ]; then
    python3 math_inference.py --mode $evaluation_mode
else
    echo "Invalid dataset specified. Choose 'mmlu' or 'math'."
    exit 1
fi
