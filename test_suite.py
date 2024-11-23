import subprocess
import os
import logging
logging.basicConfig(filename='test_suite.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# pid 1188549
parent_dir = 'data'

stackexchange_datasets = ['interpersonal','judaism','parenting','philosophy','travel','workplace','worldbuilding']
large_datasets = ['wikidetox','cockamamie','goodreads','unhealthyconversations','workplace','worldbuilding']
# method = 'prompt_tuning'
peft_or_adapter = 'peft'
use_user_profile = True
zeroshot = True
num_epochs = 8
batch_size = 8 # default 8

# TODO: RUN export PYTHONPATH=$PATHONPATH:`pwd` MANUALLY!!!!
# subprocess.run("export PYTHONPATH=$PATHONPATH:`pwd`", shell=True)
# subprocess.run(f"pip uninstall adapter-transformers", shell=True)
# subprocess.run(f"pip uninstall transformers", shell=True)
# subprocess.run(f"pip install -r requirements_{peft_or_adapter}.txt", shell=True)

# logging.info(f'Testing params: {dict(method=method,peft_or_adapter=peft_or_adapter,use_user_profile=use_user_profile,num_epochs=num_epochs,batch_size=batch_size)}')

for i in os.walk(parent_dir):
    try:
        if i[0] == parent_dir:
            continue
        dataset = i[0].replace(f'{parent_dir}/','')
        if dataset in large_datasets:
            continue
        # test
        test_command = f"python peft_u/trainer/baseline_{peft_or_adapter}.py test --dataset_name '{dataset}' --zeroshot {zeroshot} --use_user_profile {str(use_user_profile)}"
        # run
        subprocess.run(test_command, shell=True)
    except:
        logging.info(f'Failed test for {dataset}')