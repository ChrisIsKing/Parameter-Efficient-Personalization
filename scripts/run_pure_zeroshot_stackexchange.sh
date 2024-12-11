#!/bin/bash

# Default model
model="meta-llama/Llama-3.2-1B"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# List of stackexchange datasets from train_suite.py
stackexchange_datasets=("interpersonal" "judaism" "parenting" "philosophy" "travel" "workplace" "worldbuilding")

# Loop through each dataset, prepare the data, and run the zero-shot testing command
for dataset in "${stackexchange_datasets[@]}"; do
    echo "Preparing data for dataset: $dataset"
    python peft_u/write_data/prepare_stackexchange.py --substack "$dataset"
    
    echo "Running zero-shot testing for dataset: $dataset"
    python peft_u/trainer/baseline_peft.py test --dataset_name "$dataset" --zeroshot True --model "$model" --max_example_count 0
done