import csv
import json
import random
from os.path import join as os_join
from typing import Tuple, List, Dict, Union, Any
from collections import defaultdict

import nltk
from nltk.metrics.distance import masi_distance
from tqdm import tqdm

from stefutil import *
from peft_u.util import *


__all__ = [
    'PersonalizedData', 'PersonalizedDataset',
    'load_csv', 'data2dataset_splits', 'avg_num_users_per_example', 'save_datasets'
]


logger = get_logger('Convert Data Format')


PersonalizedData = Dict[Any, Dict[str, Dict[str, Union[str, List[str]]]]]  # user_id -> sample id -> a dict of sample

# user_id -> dataset split -> sample id -> a dict of sample
PersonalizedDataset = Dict[str, Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]]]


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


def _split2train_val_test(samples: List = None, train_split_ratio: float = 0.8, seed: int = 42):
    set_seed(seed)
    random.shuffle(samples)

    tr_split_sz = round(train_split_ratio * len(samples))
    tr, ts = samples[:tr_split_sz], samples[tr_split_sz:]

    # split remaining examples into test and val
    vl_split_sz = round(0.5 * len(ts))
    vl, ts = ts[vl_split_sz:], ts[:vl_split_sz]
    return tr, vl, ts


def data2dataset_splits(
        data: PersonalizedData = None, train_split_ratio: float = 0.8, seed: int = 42, leakage: bool = True
) -> Tuple[PersonalizedDataset, List, Dict[str, Dict[str, int]]]:
    """
    Split data into train, val and test sets

    :param data: data to split
    :param train_split_ratio: ratio of train set
    :param seed: random seed
    :param leakage: whether to allow text sample leakage between train and test sets across users
    :return: 3-tuple of (
        split dataset,
        agreement data on annotation,
        users with too little samples and the corresponding split sizes
    )
    """
    agreement_data = []
    user_data = defaultdict(lambda: defaultdict(dict))

    it = tqdm(data.items(), desc='Splitting data', total=len(data), unit='user')
    if leakage:
        for uid, data_ in it:
            agreement_data += [(uid, post_id, frozenset(value['label'])) for post_id, value in data_.items()]

            label_options = sorted(set().union(*[v['label'] for k, v in data_.items()]))

            tr_, vl_, ts_ = dict(), dict(), dict()  # to disable warning
            for label in label_options:
                samples = [(sid, sample) for sid, sample in data_.items() if label in sample['label']]
                tr, vl, ts = _split2train_val_test(samples, train_split_ratio=train_split_ratio, seed=seed)

                u_data = user_data[uid]
                tr_, vl_, ts_ = u_data['train'], u_data['val'], u_data['test']
                # to ensure multi-label sample appears in only one split
                tr_.update({sid: sample for (sid, sample) in tr if sid not in vl_ and sid not in ts_})
                ts_.update({sid: sample for (sid, sample) in ts if sid not in tr_ and sid not in vl_})
                vl_.update({sid: sample for (sid, sample) in vl if sid not in tr_ and sid not in ts_})
            assert set(tr_) & set(ts_) == set() and set(tr_) & set(vl_) == set() and set(ts_) & set(vl_) == set()
            assert len(tr_) + len(ts_) + len(vl_) == len(data_)  # sanity check mutually exclusive
    else:
        lst_sids = list(set([sid for uid, data_ in data.items() for sid in data_]))
        random.shuffle(lst_sids)

        tr, vl, ts = _split2train_val_test(lst_sids, train_split_ratio=train_split_ratio, seed=seed)
        tr_s, vl_s, ts_s = set(tr), set(vl), set(ts)  # for faster lookup

        for uid, data_ in it:
            # By construction, not necessary that each split contains all label options
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

    small_user_meta = {uid: u_data for uid, u_data in user_data.items() if any(len(d_) == 0 for d_ in u_data.values())}
    small_user_meta = {uid: {split: len(d) for split, d in u_data.items()} for uid, u_data in small_user_meta.items()}
    if len(small_user_meta) > 0:
        uids = sort_user_ids(list(small_user_meta.keys()))
        logger.warning(f'Users with too little samples: {pl.i(uids)}')

    return user_data, agreement_data, small_user_meta


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


def save_datasets(data: PersonalizedData = None, base_path: str = None):
    """
    Saves processed personalized dataset as json files on disk, one leaked and one non-leaked.
    """
    num_annotators = len(data)
    num_examples = sum([len(v) for k, v in data.items()])

    split_args = dict(data=data, train_split_ratio=0.8, seed=42)
    user_data_leaked, agreement_data, too_small = data2dataset_splits(**split_args, leakage=True)
    user_data_no_leak, agreement_data_, too_small_ = data2dataset_splits(**split_args, leakage=False)
    assert set(agreement_data) == set(agreement_data_)  # sanity check
    assert set(too_small.keys()) == set(too_small.keys())

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
