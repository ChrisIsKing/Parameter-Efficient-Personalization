from utils import *
import json
import nltk
from nltk.metrics.distance import masi_distance

data = json.load(open('cockamamie/cockamamie.json'))

user_data = {}

post_map = {}

for item in data['word_ratings']['votes']:
    for key, value in item.items():
        # Loop through yes and no votes
        for voter in value['no_votes']:
            if voter not in user_data:
                user_data[voter] = {}
            if key not in post_map:
                post_map[key] = len(post_map)
            user_data[voter][post_map[key]] = {"text": key, "label": ["no"]} 
        for voter in value['yes_votes']:
            if voter not in user_data:
                user_data[voter] = {}
            if key not in post_map:
                post_map[key] = len(post_map)
            user_data[voter][post_map[key]] = {"text": key, "label": ["yes"]}

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
with open('cockamamie/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('cockamamie/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)
