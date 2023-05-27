import json
from os.path import join as os_join
from collections import Counter

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


def _most_common(lst, tie_breaker: str = None):
    """
    Returns the most common element in a list.
    If the count of the top 2 elements are the same then `tie_breaker` is returned.
    """
    c = Counter(lst)
    top_2 = c.most_common(2)
    if len(top_2) > 1:
        if top_2[0][1] == top_2[1][1]:
            return tie_breaker
    return top_2[0][0]


def prepare_data(data, test_ids=None, random_state=42):
    """
    Prepare hatexplain data.
    """
    set_seed(random_state)
    ruleset = {}
    normal_examples = {}
    global_examples = {"all": {}, "train": {}, "test": {}, "val": {}}

    for k, v in data.items():
        label = _most_common(lst=[annotator['label'] for annotator in v['annotators']], tie_breaker="undecided")
        group = [
            item for item, count in
            Counter([group for annotator in v['annotators'] for group in annotator['target']]).items() if count > 1
        ]
        if not group:
            group = ["None"]
        post_text = " ".join(v['post_tokens'])
        if label == "normal":
            v['majority_label'] = ["normal"]
            v['text'] = post_text
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
            global_examples["all"][k]['majority_label'] = [label]
            global_examples["all"][k]['text'] = post_text
            global_examples["all"][k]['rationale_spans'] = []
            global_examples["all"][k]['majority_group'] = group
        for index, annotator in enumerate(v['annotators']):
            if annotator['annotator_id'] not in ruleset:
                ruleset[annotator['annotator_id']] = {}
            if v['rationales']:
                if len(v['rationales']) <= index:
                    ruleset[annotator['annotator_id']][v['post_id']] = {
                        'post_tokens': v['post_tokens'],
                        'text': post_text,
                        'rationale': [],
                        'label': v['annotators'][index]['label'],
                        'target_group': v['annotators'][index]['target'],
                        'rationale_tokens': [],
                        'rationale_spans': [],
                        'majority_label': [label],
                        'majority_group': group
                    }
                else:
                    ruleset[annotator['annotator_id']][v['post_id']] = {
                        'post_tokens': v['post_tokens'],
                        'text': post_text,
                        'rationale': v['rationales'][index],
                        'label': v['annotators'][index]['label'],
                        'target_group': v['annotators'][index]['target'],
                        'rationale_tokens': [token for i, token in enumerate(v['post_tokens']) if
                                             v['rationales'][index][i] == 1],
                        'rationale_spans': [],
                        'majority_label': [label],
                        'majority_group': group
                    }
                    queue = []
                    for flag, value in zip(v['rationales'][index], v['post_tokens']):
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
                    'text': post_text,
                    'rationale': [],
                    'label': v['annotators'][index]['label'],
                    'target_group': v['annotators'][index]['target'],
                    'rationale_tokens': [],
                    'rationale_spans': [],
                    'majority_label': [label],
                    'majority_group': group
                }

    return ruleset, normal_examples, global_examples


if __name__ == '__main__':
    def run():
        label_map = dict(normal=0, hatespeech=1, offensive=2)

        dset_base_path = os_join(u.proj_path, u.dset_dir, 'hatexplain')

        with open(os_join(dset_base_path, 'dataset.json')) as fl:
            data = json.load(fl)
        with open(os_join(dset_base_path, 'post_id_divisions.json')) as fl:
            test_ids = json.load(fl)

        user_data, normal_examples, global_examples = prepare_data(data, test_ids)

        save_datasets(data=user_data, base_path=dset_base_path)
    run()
