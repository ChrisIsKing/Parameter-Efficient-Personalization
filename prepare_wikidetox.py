from utils import *
import json
import nltk
from nltk.metrics.distance import masi_distance

annotation_data, annotation_headers = load_csv('data/wikidetox/aggression_annotations.tsv', delimiter="\t", header=True)
text_data, text_headers = load_csv('data/wikidetox/aggression_annotated_comments.tsv', delimiter="\t", header=True)

# Build id to text map
id_to_text = {}
for i, row in enumerate(text_data):
    post_id = row[0]
    text = row[1]
    id_to_text[post_id] = text

# Build user data
user_data = {}

for i, row in enumerate(annotation_data):
    post_id = row[0]
    user_id = row[1]
    label = row[2]
    if user_id not in user_data:
        user_data[user_id] = {}
    if post_id not in user_data[user_id]:
        user_data[user_id][post_id] = {"text": id_to_text[post_id], "label": label}
    
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
with open('data/wikidetox/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('data/wikidetox/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)