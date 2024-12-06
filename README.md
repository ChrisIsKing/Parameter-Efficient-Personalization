# PEFT-U: Parameter-Efficient Fine-Tuning for User Personalization at Scale

This repository consists of the code and data for the paper Parameter-Efficient Fine-Tuning for User Personalization at Scale. In this work we explore the problem of personalizing LLMs to support user centered tasks where the preference of users' can potentially differ for the same input. Conventional LLMs focus on having one model to support all users, we emprically analyze the use of prompting LLMs vs tuning and compartmentalizing user-level knowlege for personalized tasks.



Python version `3.10.9`. 





## Personalized Training

### Installation 

For PEFT methods, e.g. `LoRA`, `P-tuning` and personalized head, run 

```bash
pip install -r requirements_peft.txt
```

For Adapter methods, run 

```bash
pip install -r requirements_adapter.txt
```

Two separate environments are needed since [`adapter-transformers`](https://github.com/adapter-hub/adapter-transformers) directly modifies HuggingFace’s [`transformers`](https://github.com/huggingface/transformers) and overrides their identifiers. 

Add current directory for python to look for our local package:
    
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`
```

All scripts are intended to be run from the root directory. 

## Process datasets

Run files in `peft_u/write_data`. 

For example, process the TweetEval dataset, run 

```python
python peft_u/write_data/prepare_gabhate.py
```

## Run Training

### PEFT

**Arguments**

Shared arguments for `train` and `test` sub-parsers 

-   `model`: HuggingFace model name or local model path 
-   `dataset_name`: Dataset name 
-   `leakage`: whether to allow text sample leakage between train and test set across users 
    -   Current experiments are w/ `leakage == True`
-   `method`: PEFT frameworks, one of [`lora`, `prefix`, `p_tuning`, `prompt_tuning`]
-   `seed`: random seed for 1) loading dataset demo examples and 2) training 
-   `batch_size`: batch size for training/evaluation 

Additional arguments for the `train` sub-parser 

-   `num_epochs`: Number of training epochs 
-   `learning_rate`: Optimizer learning rate 
-   `weight_decay`: Optimizer weight decay 
-   `output_dir`: Output directory name postfix 
    -   Note an output directory name will be generated from the training arguments 

Additional arguments for the `test` sub-parser 

-   `zeroshot`: If given, a single model is loaded for inference on all users with no personalization 
    -   Intended for evaluating zero-shot performance 

**Example**

1> Run `Prefix Tuning` training with the `EPIC` dataset: 

```bash
python peft_u/trainer/baseline_peft.py train --dataset_name 'epic' --method 'prefix'
```

2> Run inference on a a model trained with `UnhealthyConversations`

```bash
python peft_u/trainer/baseline_peft.py test --dataset_name 'unhealthyconversations' --model '23-06-03_{md_nm=flan-t5-base, ds=unhealthyconversations, peft=p_tuning}'
```

### Adapter 

**Arguments**

The same as that of `PEFT`, except that 

1.   `method`: Adapter methods, one of [`Houlsby`, `IA3`]
2.   There’s no `zeroshot` flag for the `test` sub-parser 

**Example**

1> Run `IA3` training with the `TweetEval` dataset for 16 epochs, with output postfix `16ep`

```bash
python peft_u/trainer/baseline_adapter.py train --dataset_name 'tweeteval' --method 'IA3' --num_epochs 16 --output_dir '16ep'
```

2> Run inference on a model trained with the `WikiDetox`  dataset 

```bash
python peft_u/trainer/baseline_adapter.py test --dataset_name 'wikidetox' --model '23-06-22_{adapter=Houlsby, md_nm=flan-t5-base, ds=wikidetox}'
```

### Personalized Head

 **Arguments**

The same as that of `PEFT`, except that there’s no `method` argument, for the only method is `Personalized Head`. 

**Example**

1> Run `Personalized Head` training with the `StudEmo` dataset: 

```bash
python peft_u/trainer/personalized_head.py train --dataset_name 'studemo'
```

2> Run inference on a model trained with the `TweetEval` dataset 

```bash
python peft_u/trainer/personalized_head.py test --dataset_name "tweeteval" --model "23-08-05_{PH, md_nm=flan-t5-base, ds=tweeteval}"
```