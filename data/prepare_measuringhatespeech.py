from utils import *
import datasets
import json
import nltk
from nltk.metrics.distance import masi_distance
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
df = dataset['train'].to_pandas()

user_data = {}

for i in range(len(df)):
    post_id = str(df['comment_id'][i])
    user_id = str(df['annotator_id'][i])
    text = df['text'][i]
    label = str(int(df['hatespeech'][i]))

    if user_id not in user_data:
        user_data[user_id] = {}
    
    user_data[user_id][post_id] = {'text': text, 'label': [label]}

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
with open('measuringhatespeech/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('measuringhatespeech/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)