from utils import *
import json
import nltk
from nltk.metrics.distance import masi_distance

data, headers = load_csv('gabhate/GabHateCorpus_annotations.tsv', delimiter="\t", header=True)

label_map = {0: 'Non-hateful', 1: 'Hateful'}

user_data = {}

for row in data:
    post_id = row[0]
    user_id = row[1]
    text = row[2]
    label = label_map[int(row[3])]

    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id][post_id] = {"text": text, "label": [label]}

num_annotators = len(user_data)
num_examples = sum([len(v) for k, v in user_data.items()])

user_data_leaked, agreement_data = data2dataset_splits(user_data, 0.8, seed=42, leakage=True)
user_data_no_leak, agreement_data = data2dataset_splits(user_data, 0.8, seed=42, leakage=False)

masi_task = nltk.AnnotationTask(distance=masi_distance)
masi_task.load_array(agreement_data)
print("Krippendorff's alpha: {}".format(masi_task.alpha()))
print("Number of Users: {}".format(num_annotators))
print("Number of Examples: {}".format(num_examples))
print("Average number of examples per user: {}".format(num_examples/num_annotators))
print("Average number of users per example: {}".format(avg_num_users_per_example(user_data)))

# Save the data
with open('gabhate/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('gabhate/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)
