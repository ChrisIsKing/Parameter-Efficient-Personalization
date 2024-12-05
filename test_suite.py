import subprocess
import os
import logging
import re
import json
logging.basicConfig(filename='test_suite.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# pid 1188549
model_dir = '/data/Parameter-Efficient-Personalization/models'

stackexchange_datasets = ['interpersonal','judaism','parenting','philosophy','travel','workplace','worldbuilding']
large_datasets = ['wikidetox','cockamamie','measuringhatespeech','unhealthyconversations','workplace','worldbuilding','goodreads']
# method = 'lora'
peft_or_adapter = 'peft'
# use_user_profile = True
zeroshot = True
# num_epochs = 8
# batch_size = 8 # default 8

datasets = stackexchange_datasets + ['goodreads']

for dataset in datasets:
    suffix = ""
    if dataset in large_datasets:
        suffix = "--num_samples 100"
    if dataset in stackexchange_datasets:
        prep_command = f"python3 peft_u/write_data/prepare_stackexchange.py --substack '{dataset}' {suffix}"
    else:
        prep_command = f"python3 peft_u/write_data/prepare_goodreads.py {suffix}"
    # print(prep_command)
    subprocess.run(prep_command, shell=True)

for up in [True,False]:
    for dataset in datasets:
        try:
            # test
            test_command = f"python peft_u/trainer/baseline_{peft_or_adapter}.py test --dataset_name '{dataset}' --zeroshot {zeroshot} --use_user_profile {up} --model 'meta-llama/Llama-3.2-1B'"
            # run
            # print(test_command)
            subprocess.run(test_command, shell=True)
        except:
            logging.info(f'Failed test for {dataset}')