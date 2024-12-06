import pandas as pd
import os
import re
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S"  # Set the date format
)

parent_dir = 'eval/'

train_date_pattern = r"^(\d{2}-\d{2}-\d{2})_"
model_dict_pattern = r"\{(.*?)\}"
test_date_pattern = r"Eval-(\d{2}-\d{2}-\d{2})$"

file_pattern = r"test_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.log"

generative_datasets = ['interpersonal','judaism','parenting','philosophy','travel','workplace','worldbuilding','goodreads']

full_dict = {}
for i in os.walk(parent_dir):
    if i[0] == parent_dir:
        continue

    try:
        # Extract values
        # print(subdir_str)
        # train_date = re.search(train_date_pattern, subdir_str).group(1)
        subdir = i[0].replace(parent_dir,'')
        foo = re.sub(r'(\w+)=([^,}]+)', r'"\1": "\2"', re.search(model_dict_pattern, subdir).group(1))
        model_dict = json.loads(f"{{{foo}}}")
        model_dict['train_date'] = re.search(train_date_pattern, subdir).group(1)
        model_dict['test_date'] = re.search(test_date_pattern, subdir).group(1)
        model_dict['machine'] = 'clarity2'

        test_files = [f for f in os.listdir(parent_dir+subdir) if re.match(file_pattern, f)]
        if len(test_files) == 0:
            logging.warning(f"No test results for {subdir}")
            continue
        test_files.sort(key=lambda x: datetime.strptime(re.match(file_pattern, x).group(1), "%Y-%m-%d_%H-%M-%S"), reverse=True)
        test_file_selected = test_files[0] # use most recent

        if not 'test_date' in model_dict.keys() or model_dict['test_date'] < '24-12-04':
            continue
        # if 'md' in model_dict.keys() and model_dict['md'] != 'Llama-3.2-1B':
        #     continue

        # Determine metric type
        if model_dict['ds'] in generative_datasets:
            model_dict['metric'] = 'rouge'        
            # Read and extract the score
            file_path = os.path.join(parent_dir+subdir, test_file_selected)
            # print(file_path)
            with open(file_path, 'r') as file:
                content = file.read()
                content = content.split("macro-avg rouge:",1)[1]
                model_dict['score'] = content[:content.rfind('}')].replace('\n','')
        else:
            model_dict['metric'] = 'accuracy'
            file_path = os.path.join(parent_dir+subdir, test_file_selected)
            # print(file_path)
            with open(file_path, 'r') as file:
                content = file.read()
                model_dict['score'] = re.search(r"macro-avg acc: (\d+\.?\d*)", content).group(1)
        full_dict[i[0]] = model_dict
            # print(test_date)
            # Print extracted values
            # logging.info(f"Train Date: {train_date}")
            # logging.info("Model Dict:", model_dict)
            # logging.info("Test Date:", test_date)
    except:
        logging.info(f'Skipping: {i[0]}',)
        # print('skip')
        continue

df = pd.DataFrame.from_dict(full_dict, orient='index').reset_index().rename(columns={'index':'path'})
df.to_csv('results.csv')