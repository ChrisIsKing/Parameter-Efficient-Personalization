import csv
import json
import random
from os.path import join as os_join
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Union, Any

import nltk
from nltk.metrics.distance import masi_distance
from tqdm import tqdm

from stefutil import *
from peft_u.util import *


__all__ = [
    'instructions', 'InputEgDataset', 'process_data',
    'load_csv', 'data2dataset_splits', 'avg_num_users_per_example',
    'PersonalizedDataset', 'save_datasets'
]


logger = get_logger('Data Utils')


PersonalizedDataset = Dict[Any, Dict[str, Dict[str, Union[str, List[str]]]]]  # user_id -> sample id -> a dict of sample


instructions = {
    "cockamanie":
        "Please rate whether the following text is funny or not funny. Respond 'yes' if you think the text is funny "
        "and 'no' if you think the text is not funny.",
    "gabhate":
        "Please review the following text and indicate if it has the presence of hate speech. "
        "Respond 'Hateful' if the text contains hate speech "
        "and 'Non-hateful' if the text does not contain hate speech.",
    "goemotion":
        "Please analyze the following text and assign one or more appropriate emotion labels. "
        "Emotion labels include happiness, sadness, anger, surprise, joy, fear, disgust. "
        "You can select one or multiple emotion labels that best capture the emotional content of the text. "
        "Respond with the emotion labels separated by a comma.",
    "hatexplain":
        "Please review the following text "
        "and indicate if it has the presence of hate speech or is offensive in nature. "
        "Respond 'hatespeech' if the text contains hate speech, "
        "'offensive' if the text does not contain hate speech but is offensive "
        "and 'normal' if the text is neither offensive nor contains hate speech.",
    "measuringhatespeech":
        "On a scale of 0-2 rate the level of hatespeech contained in the following text. "
        "Respond with '0' if the text does not contain hate speech, "
        "'1' if the text contains some hate speech, and '2' if the text contains a lot of hate speech.",
    "studemo":
        "Please analyze the following text and assign one or more appropriate emotion labels. "
        "Emotion labels include "
        "joy, trust, anticipation, surprise, fear, sadness, disgust, anger, valence, and arousal. "
        "You can select one or multiple emotion labels that best capture the emotional content of the text. "
        "Respond with the emotion labels separated by a comma.",
    "subjectivediscourse_response":
        "Please analyze the following text and indicate how the witness responded to the question. "
        "Respond with 'answer' if they answered the question reasonably, "
        "'cant-answer-lying' if they could not answer and are lying, "
        "'can't answer-sincere' if they could not answer but are honest about it, "
        "'shift-dodge' if they shifted the topic with the intent of dodging the question, "
        "'answer_overans-sway' if they over answered the question with the intention of swaying "
        "or 'shift-correct' if they shifted the topic with the intention of clarifying the question.",
    "subjectivediscourse_question_sentiment":
        "Please analyze the following text and rate your sentiment towards the questioners. "
        "Sentiment labels include "
        "'somewhatPositive', 'positive', 'veryPositive', 'somewhatNegative', 'veryNegative', 'neutral' and 'negative'. "
        "Respond with the sentiment label that best captures your sentiment towards the questioners.",
    "subjectivediscourse_response_sentiment":
        "Please analyze the following text and rate your sentiment towards the witness. "
        "Sentiment labels include "
        "'somewhatPositive', 'positive', 'veryPositive', 'somewhatNegative', 'veryNegative', 'neutral' and 'negative'. "
        "Respond with the sentiment label that best captures your sentiment towards the witness.",
    "tweeteval":
        "Please review the following text and indicate if it has the presence of hate speech. "
        "Respond 'Hateful' if the text contains hate speech "
        "and 'Non-hateful' if the text does not contain hate speech.",
    "unhealthyconversations":
        "Please review the following text and indicated if it is 'healthy' or 'unhealthy'. "
        "Respond 'healthy' if the text is healthy "
        "and 'unhealthy' "
        "if the text can be considered hostile, antagonistic, condescending, dismissive or an unfair generalization.",
    "wikidetox":
        "Please review the following text "
        "and indicate if it has the presence of malicious remark to a person or group. "
        "Respond 'Aggressive' if the text contains a personal attack "
        "and 'Normal' if the text does not contain a personal attack.",
}


class InputExample:
    def __init__(self, guid: Union[int, str], instruction: str = None, text: str = None, prompt_examples: List = None,
                 label: List[str] = None) -> None:
        if isinstance(guid, str):
            guid = int(guid)
        self.guid = guid
        self.text = text
        self.prompt_examples = prompt_examples
        self.instruction = instruction
        self.label = label

    def process_template(self):
        prompt = f"{self.instruction} "
        # TODO: separator between examples?
        for example in self.prompt_examples:
            txt, lb = example  # TODO: consider multiple examples?
            prompt += f"Text: {txt} Label: {lb}. "

        prompt += f"Text: {self.text} Label: "

        return prompt

    def process_target(self):
        if len(self.label) == 1:
            return self.label[0]
        return ','.join(self.label)

    def __repr__(self):
        return f"InputExample(guid={self.guid}, instruction={self.instruction}, text={self.text}, " \
               f"prompt_examples={self.prompt_examples}, label={self.label})"


@dataclass
class InputEgDataset:
    train: List[InputExample]
    val: List[InputExample]
    test: List[InputExample]


def process_data(
        data: Dict[str, Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]]], task: str,
        example_count: int = 1, max_example_count: int = 3, per_user: bool = True
) -> Union[InputEgDataset, Dict[str, InputEgDataset]]:
    """
    Process data for few-shot learning

    :param data: dataset of user id => dataset split => sample id => sample data
    :param task: dataset name (e.g. goemotion)
    :param example_count: number of examples in few-shot prompt, per category
    :param max_example_count: maximum number of examples to use per category
    :param per_user: If True, return datasets for each user
    """
    ret = dict()
    instruction = instructions[task]

    for uid, dset in data.items():
        def split2label_options(split: str) -> List[str]:
            return sorted(set().union(*[v['label'] for k, v in dset[split].items()]))

        # Get all labels in the train split
        label_options = split2label_options('train')
        # sanity check, TODO: maybe not the case?
        assert label_options == split2label_options('val') == split2label_options('test')
        lb2txts: Dict[str, List[str]] = {  # label => list of examples w/ that label
            label: [sample['text'] for id_, sample in dset['train'].items() if label in sample['label']]
            for label in label_options
        }

        def get_split(split: str) -> List[InputExample]:
            """
            Add prompt example to each sample in the split
            """
            lst = []
            for sid, sample in dset[split].items():
                text, label = sample['text'], sample['label']

                # take n random examples from each category for few-shot prompt
                prompt_egs = [  # TODO: possible to select the exact same `text` in the prompt?
                    [(txt, lb) for txt in random.sample(lb2txts[lb], k=min(example_count, len(lb2txts[lb])))]
                    for lb in label_options
                ]
                prompt_egs = sum(prompt_egs, start=[])[:max_example_count]

                lst.append(
                    InputExample(guid=sid, instruction=instruction, text=text, prompt_examples=prompt_egs, label=label)
                )
            return lst
        ret[uid] = InputEgDataset(
            train=get_split('train'),
            val=get_split('val'),
            test=get_split('test')
        )
    if per_user:
        return ret
    else:
        return InputEgDataset(
            train=sum([dset.train for dset in ret.values()], start=[]),
            val=sum([dset.val for dset in ret.values()], start=[]),
            test=sum([dset.test for dset in ret.values()], start=[])
        )


def load_csv(path, delimiter=",", header=True):
    """
    Load a csv file and return header as dict and data.
    """
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            header = next(reader)
            header = {i: header[i] for i in range(len(header))}
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


def split2train_val_test(samples: List = None, train_split_ratio: float = 0.8, seed: int = 42):
    set_seed(seed)
    random.shuffle(samples)

    tr_split_sz = int(train_split_ratio * len(samples))
    tr, ts = samples[:tr_split_sz], samples[tr_split_sz:]

    # split remaining examples into test and val
    vl_split_sz = int(0.5 * len(ts))
    vl, ts = ts[vl_split_sz:], ts[:vl_split_sz]
    return tr, vl, ts


def data2dataset_splits(
        data: PersonalizedDataset, train_split_ratio: float = 0.8, seed: int = 42, leakage: bool = True
):
    """
    Split data into train, val and test sets

    :param data: data to split
    :param train_split_ratio: ratio of train set
    :param seed: random seed
    :param leakage: whether to allow text sample leakage between train and test sets across users
    """
    agreement_data = []
    user_data = defaultdict(lambda: defaultdict(dict))

    if leakage:
        for uid, data_ in tqdm(data.items(), desc='Splitting data', total=len(data)):
            agreement_data += [(uid, post_id, frozenset(value['label'])) for post_id, value in data_.items()]

            label_options = sorted(set().union(*[v['label'] for k, v in data_.items()]))
            # mic(label_options, label_options)
            # raise NotImplementedError

            # tr_, vl_, ts_ = dict(), dict(), dict()
            for label in label_options:
                samples = [(sid, sample) for sid, sample in data_.items() if label in sample['label']]
                tr, vl, ts = split2train_val_test(samples, train_split_ratio=train_split_ratio, seed=seed)

                u_data = user_data[uid]
                tr_, vl_, ts_ = u_data['train'], u_data['val'], u_data['test']
                # TODO: here, all the label for the given text is stored,
                #  should also check not already in current split?
                tr_.update({sid: sample for (sid, sample) in tr if sid not in ts_})
                ts_.update({sid: sample for (sid, sample) in ts if sid not in tr_})
                vl_.update({sid: sample for (sid, sample) in vl if sid not in tr_ and sid not in ts_})

            # TODO: this is not the case, the same text may appear more then once for the same user in the same split?
            # assert set(tr_) & set(ts_) == set() and set(tr_) & set(vl_) == set() and set(ts_) & set(vl_) == set()
            # assert len(tr_) + len(ts_) + len(vl_) == len(data_)  # sanity check mutually exclusive
            # raise NotImplementedError
        return user_data, agreement_data

    else:
        lst_sids = list(set([sid for uid, data_ in data.items() for sid in data_]))
        random.shuffle(lst_sids)

        tr, vl, ts = split2train_val_test(lst_sids, train_split_ratio=train_split_ratio, seed=seed)
        tr_s, vl_s, ts_s = set(tr), set(vl), set(ts)  # for faster lookup

        for uid, data_ in tqdm(data.items(), desc='Splitting data', total=len(data)):
            u_data = user_data[uid]
            tr_, vl_, ts_ = u_data['train'], u_data['val'], u_data['test']

            for post_id, value in data_.items():
                agreement_data.append((uid, post_id, frozenset(value['label'])))

                if post_id in tr_s:
                    d = tr_
                elif post_id in ts_s:
                    d = ts_
                else:
                    assert post_id in vl_s  # sanity check
                    d = vl_
                d[post_id] = value
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


def save_datasets(data: PersonalizedDataset = None, base_path: str = None):
    """
    Saves processed personalized dataset as json files on disk, one leaked and one non-leaked.
    """
    num_annotators = len(data)
    num_examples = sum([len(v) for k, v in data.items()])

    user_data_leaked, agreement_data = data2dataset_splits(data, 0.8, seed=42, leakage=True)
    user_data_no_leak, agreement_data_ = data2dataset_splits(data, 0.8, seed=42, leakage=False)
    assert set(agreement_data) == set(agreement_data_)  # sanity check

    masi_task = nltk.AnnotationTask(distance=masi_distance)
    masi_task.load_array(agreement_data)
    d_log = {
        'Krippendorff\'s alpha': masi_task.alpha(),
        '#Users': num_annotators,
        '#Examples': num_examples,
        'Avg #Examples/User': num_examples/num_annotators,
        'Avg #Users/Example': avg_num_users_per_example(data),
    }
    logger.info(pl.i(d_log))

    # Save the data
    fnm_leaked = os_join(base_path, 'user_data_leaked.json')
    with open(fnm_leaked, 'w') as f:
        json.dump(user_data_leaked, f)
    logger.info(f'Leaked data saved to {pl.i(fnm_leaked)}')

    fnm_no_leak = os_join(base_path, 'user_data_no_leak.json')
    with open(fnm_no_leak, 'w') as f:
        json.dump(user_data_no_leak, f)
    logger.info(f'Non-leaked data saved to {pl.i(fnm_no_leak)}')


if __name__ == '__main__':
    def check_process_data():
        dnm = 'tweeteval/user_data_leaked.json'
        data_path = os_join(u.proj_path, 'data', dnm)
        with open(data_path, 'r') as f:
            data = json.load(f)

        def check_data_fmt():
            users = list(data.keys())
            mic(users)
            tr_u1 = data[users[0]]['train']
            mic(len(tr_u1))
            mic(tr_u1)
        # check_data_fmt()

        dset = process_data(data, task='tweeteval')
        mic(dset.keys())
    check_process_data()
