from utils import *
import json
import nltk
from nltk.metrics.distance import masi_distance

label_map = json.load(open('goemotion/ekman_mapping.json'))
label_map = flip_dict_of_lists(label_map)
label_map['neutral'] = 'neutral'

csv_files = [
    'goemotion/goemotions_1.csv',
    'goemotion/goemotions_2.csv',
    'goemotion/goemotions_3.csv',
]

user_data = {}
for csv_file in csv_files:
    data, header = load_csv(csv_file, delimiter=",", header=True)
    for row in data:
        text = row[0]
        post_id = row[1]
        user_id = row[7]
        label = list(set([label_map[header[i+9]] for i, item in enumerate(row[9:]) if item == '1']))
        if "neutral" in label:
            continue
        if len(label) < 1:
            continue

        if user_id not in user_data:
            user_data[user_id] = {}
        if post_id not in user_data[user_id]:
            user_data[user_id][post_id] = {"text": text, "label": label}

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
with open('goemotion/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('goemotion/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)