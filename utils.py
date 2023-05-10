import os
import csv
import random
import numpy as np
from collections import Counter

def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

def flip_dict_of_lists(d):
    """
    Flip a dictionary of lists.
    """
    return {v: k for k, values in d.items() for v in values}

def load_csv(path, delimiter=",", header=True):
    """
    Load a csv file and return header as dict and data.
    """
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            header = next(reader)
            header = {i:header[i] for i in range(len(header))}
        for row in reader:
            data.append(row)
    return data, header

def most_common(lst, tie_breaker=None):
    """
    Returns the most common element in a list.
    If the count of the top 2 elements are the same then  tie_breaker is returned.
    """
    data = Counter(lst)
    top_2 = data.most_common(2)
    if len(top_2) > 1:
        if top_2[0][1] == top_2[1][1]:
            return tie_breaker
    return top_2[0][0]

def get_current_directory():
    """
    Get the current directory.
    """
    return os.path.dirname(os.path.abspath(__file__))

def prepare_data(data, train_split, test_ids=None, random_state=42):
    """
    Prepare hatexplain data.
    """
    set_seed(random_state)
    ruleset = {}
    normal_examples = {}
    global_examples = {"all": {}, "train": {}, "test": {}, "val": {}}

    for k, v in data.items():
        label = most_common(lst=[annotator['label'] for annotator in v['annotators']], tie_breaker="undecided")
        group = [item for item, count in Counter([group for annotator in v['annotators'] for group in annotator['target']]).items() if count > 1]
        if not group:
            group = ["None"]
        post_text = " ".join(v['post_tokens'])
        if label == "normal":
            v['majority_label'] = "normal"
            v['post_text'] = post_text
            v['majority_group'] = group
            normal_examples[k] = v
            v['rationale_spans'] = []
            if test_ids:
                if v['post_id'] in test_ids["train"]:
                    global_examples["train"][k] = v
                elif v['post_id'] in test_ids["test"]:
                    global_examples["test"][k] = v
                elif v['post_id'] in test_ids["val"]:
                    global_examples["val"][k] = v
        elif label != "undecided":
            global_examples["all"][k] = v
            global_examples["all"][k]['majority_label'] = label
            global_examples["all"][k]['post_text'] = post_text
            global_examples["all"][k]['rationale_spans'] = []
            global_examples["all"][k]['majority_group'] = group
        for index, annotator in enumerate(v['annotators']):
            if annotator['annotator_id'] not in ruleset:
                ruleset[annotator['annotator_id']] = {}
            if v['rationales']:
                if len(v['rationales']) <= index:
                    ruleset[annotator['annotator_id']][v['post_id']] = {
                    'post_tokens': v['post_tokens'],
                    'post_text': post_text,
                    'rationale': [],
                    'label': v['annotators'][index]['label'],
                    'target_group': v['annotators'][index]['target'],
                    'rationale_tokens': [],
                    'rationale_spans': [],
                    'majority_label': label,
                    'majority_group': group
                    }
                else:
                    ruleset[annotator['annotator_id']][v['post_id']] = {
                        'post_tokens': v['post_tokens'],
                        'post_text': post_text,
                        'rationale': v['rationales'][index],
                        'label': v['annotators'][index]['label'],
                        'target_group': v['annotators'][index]['target'],
                        'rationale_tokens': [token for i, token in enumerate(v['post_tokens']) if v['rationales'][index][i] == 1],
                        'rationale_spans': [],
                        'majority_label': label,
                        'majority_group': group
                    }
                    queue = []
                    for flag,value in zip(v['rationales'][index], v['post_tokens']):
                        if flag:
                            queue.append(value)
                        elif queue:
                            rationale = " ".join(queue)
                            ruleset[annotator['annotator_id']][v['post_id']]['rationale_spans'].append(rationale)
                            global_examples["all"][k]['rationale_spans'].append(rationale)
                            queue = []
                    if queue:
                        rationale = " ".join(queue)
                        ruleset[annotator['annotator_id']][v['post_id']]['rationale_spans'].append(rationale)
                        global_examples["all"][k]['rationale_spans'].append(rationale)
            else:
                ruleset[annotator['annotator_id']][v['post_id']] = {
                    'post_tokens': v['post_tokens'],
                    'post_text': post_text,
                    'rationale': [],
                    'label': v['annotators'][index]['label'],
                    'target_group': v['annotators'][index]['target'],
                    'rationale_tokens': [],
                    'rationale_spans': [],
                    'majority_label': label,
                    'majority_group': group
                }
    
    return ruleset, normal_examples, global_examples


def split_data(data, train_split, random_state=42, leakage=True):
    """
    Split data into train and test.
    """
    set_seed(random_state)
    agreement_data = []
    user_data = {}
    
    if leakage:
        for k, v in data.items():
            user_data[k] = {"train": {}, "test": {}, "val": {}}
            for post_id, value in v.items():
                agreement_data.append((k, post_id, frozenset(value['label'])))
            
            # Split the data into train, test, and val
            categories = list(dict.fromkeys([target for id, example in v.items() for target in example['label']]))
            for category in categories:
                category_examples = [(id, example) for id, example in v.items() if category in example['label']]
                random.shuffle(category_examples)
                split = int(train_split * len(category_examples))
                train = category_examples[:split]
                # split remaining examples into test and val
                test = category_examples[split:]
                split = int(0.5 * len(test))
                val = test[split:]
                test = test[:split]
                user_data[k]['train'] = {**user_data[k]['train'] , **{item[0]:item[1] for item in train if item[0] not in user_data[k]['test']}}
                user_data[k]['test'] = {**user_data[k]['test'] , **{item[0]:item[1] for item in test if item[0] not in user_data[k]['train']}}
                user_data[k]['val'] = {**user_data[k]['val'] , **{item[0]:item[1] for item in val if item[0] not in user_data[k]['train'] and item[0] not in user_data[k]['test']}}
        
        return user_data, agreement_data
    
    else:
        all_data = list(set([post_id for k, v in data.items() for post_id in v]))
        random.shuffle(all_data)
        split = int(train_split * len(all_data))
        train = all_data[:split]
        # split remaining examples into test and val
        test = all_data[split:]
        split = int(0.5 * len(test))
        val = test[split:]
        test = test[:split]

        for k, v in data.items():
            user_data[k] = {"train": {}, "test": {}, "val": {}}
            for post_id, value in v.items():
                agreement_data.append((k, post_id, frozenset(value['label'])))
                if post_id in train:
                    user_data[k]['train'][post_id] = value
                elif post_id in test:
                    user_data[k]['test'][post_id] = value
                elif post_id in val:
                    user_data[k]['val'][post_id] = value
        
        return user_data, agreement_data


def avg_num_users_per_example(user_data):
    """
    Compute the average number of users per text.
    """
    num_users_per_example = {}
    for user_id, user_data in user_data.items():
        for post_id, post_data in user_data.items():
            if post_id not in num_users_per_example:
                num_users_per_example[post_id] = 0
            num_users_per_example[post_id] += 1
    
    return sum(num_users_per_example.values()) / len(num_users_per_example)