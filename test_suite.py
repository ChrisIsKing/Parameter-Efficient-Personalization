import subprocess
import os
import logging
import re
import json
logging.basicConfig(filename='test_suite.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# pid 1188549
model_dir = '/data/Parameter-Efficient-Personalization/models'

# stackexchange_datasets = ['interpersonal','judaism','parenting','philosophy','travel','workplace','worldbuilding']
# large_datasets = ['wikidetox','cockamamie','goodreads','unhealthyconversations','workplace','worldbuilding']
method = 'lora'
peft_or_adapter = 'peft'
# use_user_profile = True
# zeroshot = False
# num_epochs = 8
# batch_size = 8 # default 8

train_date_pattern = r"^(\d{2}-\d{2}-\d{2})_"
model_dict_pattern = r"\{(.*?)\}"
# test_date_pattern = r"Eval-(\d{2}-\d{2}-\d{2})$"

# file_pattern = r"test_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.log"

# TODO: RUN export PYTHONPATH=$PATHONPATH:`pwd` MANUALLY!!!!
# subprocess.run("export PYTHONPATH=$PATHONPATH:`pwd`", shell=True)
# subprocess.run(f"pip uninstall adapter-transformers", shell=True)
# subprocess.run(f"pip uninstall transformers", shell=True)
# subprocess.run(f"pip install -r requirements_{peft_or_adapter}.txt", shell=True)

# logging.info(f'Testing params: {dict(method=method,peft_or_adapter=peft_or_adapter,use_user_profile=use_user_profile,num_epochs=num_epochs,batch_size=batch_size)}')
models = [i for i in next(os.walk(model_dir))[1]]
# print(models)
# exit(0)
for model in models:
    try:
        # if i[0] == model_dir:
        #     continue
        # model = i[0].replace(f'{model_dir}/','')
        # if model in large_datasets:
        #     continue

        foo = re.sub(r'(\w+)=([^,}]+)', r'"\1": "\2"', re.search(model_dict_pattern, model).group(1))
        model_dict = json.loads(f"{{{foo}}}")

        # print(model)
        # if not zeroshot:
        # model_dict['test_date'] = re.search(test_date_pattern, model).group(1)
        model_dict['train_date'] = re.search(train_date_pattern, model).group(1)

        if model_dict['train_date'] < "24-11-25":
            continue
        # else:
        #     model_dict['test_date'] = re.search(train_date_pattern, model).group(1)

        # test
        test_command = f"python peft_u/trainer/baseline_{peft_or_adapter}.py test --dataset_name '{model_dict['ds']}' --use_user_profile {model_dict['up']} --model '{model}'"
        # run
        # print(test_command)
        subprocess.run(test_command, shell=True)
    except:
        logging.info(f'Failed test for {model}')