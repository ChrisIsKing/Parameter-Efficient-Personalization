import subprocess
import os
import logging
logging.basicConfig(filename='train_suite.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# pid 1188549
parent_dir = 'data'

stackexchange_datasets = ['interpersonal','judaism','parenting','philosophy','travel','workplace','worldbuilding']
large_datasets = ['measuringhatespeech','unhealthyconversations','wikidetox','cockamamie','workplace','worldbuilding','goodreads']
methods = ['prefix','p_tuning','prompt_tuning','lora']
peft_or_adapter = 'peft'
use_user_profile = False
num_epochs = 8
batch_size = 8 # default 8
num_samples = 100
seed = 42

completed_datasets = []#'studemo','tweeteval','gabhate']#['workplace','worldbuilding','goodreads']

# TODO: RUN export PYTHONPATH=$PATHONPATH:`pwd` MANUALLY!!!!
# subprocess.run("export PYTHONPATH=$PATHONPATH:`pwd`", shell=True)
# subprocess.run(f"pip uninstall adapter-transformers", shell=True)
# subprocess.run(f"pip uninstall transformers", shell=True)
# subprocess.run(f"pip install -r requirements_{peft_or_adapter}.txt", shell=True)

# logging.info(f'Training all {peft_or_adapter} methods for: {dict(datasets=train_datasets,use_user_profile=use_user_profile,num_epochs=num_epochs,batch_size=batch_size)}')

# for dataset in stackexchange_datasets + ['goodreads']:
#     try:
#         # prep dataset
#         suffix = ""
#         if dataset in large_datasets:
#             suffix = f"--num_samples {num_samples} --seed {seed}"
#         if dataset in stackexchange_datasets:
#             prepare_command = f"python3 peft_u/write_data/prepare_stackexchange.py --substack {dataset} {suffix}"
#         else:
#             prepare_command = f"python3 peft_u/write_data/prepare_{dataset}.py {suffix}"
#         # run
#         subprocess.run(prepare_command, shell=True)
#     except:
#         logging.info(f'Failed prep for {dataset}')

for dataset in stackexchange_datasets + ['goodreads']:
    # if i[0] in parent_dir:
    #     continue
    # dataset = i[0].replace(f'{parent_dir}/','')
    # if dataset in completed_datasets:
    #     continue
    for method in methods:
        try:
            # train
            train_command = f"python peft_u/trainer/baseline_{peft_or_adapter}.py train --dataset_name '{dataset}' --method '{method}' --num_epochs {num_epochs} --use_user_profile {str(use_user_profile)} --batch_size {batch_size}"
            # run
            # print(train_command)
            subprocess.run(train_command, shell=True)
        except:
            logging.info(f'Failed train for {dataset} using {method}')

# python peft_u/trainer/baseline_peft.py train --dataset_name 'unhealthyconversations' --method 'p_tuning' --num_epochs 8 --use_user_profile False --batch_size 8