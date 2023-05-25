from utils import *
import json
import nltk
import ast
from nltk.metrics.distance import masi_distance

data, headers = load_csv('subjectivediscourse/with_features_annotated_questions_responses_gold.tsv', delimiter="\t", header=True)

user_data = {}

label_set = set()
sentiment_set = set()

for row in data:
    post_id = row[3]
    users = ast.literal_eval(row[17])
    labels = ast.literal_eval(row[11])
    q_sentiments = ast.literal_eval(row[21])
    r_sentiments = ast.literal_eval(row[18])
    text = '%s \n %s \n %s \n %s' % (row[5], row[6], row[7], row[8])

    for label in labels:
        label_set.add(label)
    
    for sentiment in q_sentiments:
        sentiment_set.add(sentiment)

    for sentiment in r_sentiments:
        sentiment_set.add(sentiment)

    for i, user in enumerate(users):
        if user not in user_data:
            user_data[user] = {}

        user_data[user][post_id] = {"text": text, "label": [labels[i]], "q_sentiment": [q_sentiments[i]], "r_sentiment": [r_sentiments[i]]}


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
print("Label Set: {}".format(label_set))
print("Sentiment Set: {}".format(sentiment_set))

# Save the data
with open('subjectivediscourse/user_data_leaked.json', 'w') as f:
    json.dump(user_data_leaked, f)

with open('subjectivediscourse/user_data_no_leak.json', 'w') as f:
    json.dump(user_data_no_leak, f)