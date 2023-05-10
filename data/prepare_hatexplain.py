from utils import *
import json
import numpy as np
import nltk
from nltk.metrics.distance import masi_distance

label_map = {"normal": 0, "hatespeech": 1, "offensive": 2}

# Load HateXplain dataset
dir = get_current_directory()
data = json.load(open(dir+'/hatexplain/dataset.json'))
test_ids = json.load(open(dir+'/hatexplain/post_id_divisions.json'))

# Prepare data for training
train_split = 0.8
user_data, normal_examples, global_examples = prepare_data(data, train_split, test_ids)

num_annotators = len(user_data)
num_examples = sum([len(v) for k, v in user_data.items()])

user_data_leaked, agreement_data = split_data(user_data, 0.8, random_state=42, leakage=True)
user_data_no_leak, agreement_data = split_data(user_data, 0.8, random_state=42, leakage=False)

masi_task = nltk.AnnotationTask(distance=masi_distance)
masi_task.load_array(agreement_data)
print("Krippendorff's alpha: {}".format(masi_task.alpha()))
print("Number of Users: {}".format(num_annotators))
print("Number of Examples: {}".format(num_examples))
print("Average number of examples per user: {}".format(num_examples/num_annotators))
print("Average number of users per example: {}".format(avg_num_users_per_example(user_data)))

# Save the data
with open('hatexplain/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('hatexplain/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)
            
