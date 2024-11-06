import re
import json
import random
from os.path import join as os_join
from typing import List, Dict, Union, Any, Callable
from dataclasses import dataclass
import csv

from torch.utils.data import Dataset

from stefutil import *
from peft_u.util import *


__all__ = [
    'PersonalizedData', 'PersonalizedDataset',
    'InputExample', 'InputEgDataset', 'ListDataset', 'get_dataset_sizes',
    'load_dataset_with_prompts', 'sort_user_ids', 'iter_users'
]


logger = get_logger('Load Dataset')


PersonalizedData = Dict[Any, Dict[str, Dict[str, Union[str, List[str]]]]]  # user_id -> sample id -> a dict of sample

# user_id -> dataset split -> sample id -> a dict of sample
PersonalizedDataset = Dict[str, Dict[str, Dict[str, Dict[str, Union[str, List[str]]]]]]


class InputExample:
    def __init__(self, guid: Union[int, str], instruction: str = None, text: str = None, prompt_examples: List = None,
                 label: List[str] = None, user_profile: str = None) -> None:
        self.guid = guid
        self.text = text
        self.prompt_examples = prompt_examples
        self.instruction = instruction
        self.user_profile = user_profile
        assert isinstance(label, list) and all(isinstance(l, str) for l in label)
        self.label = label

    def process_template(self) -> str:
        prompt = f"{self.instruction} "
        if self.user_profile:
            prompt += f"User profile: {self.user_profile}"
        for example in self.prompt_examples:
            txt, lb = example
            prompt += f"Text: {txt} Label: {lb}. "

        prompt += f"Text: {self.text} Label: "
        return prompt

    def process_target(self) -> str:
        if len(self.label) == 1:
            return self.label[0]
        return ', '.join(self.label)

    def __repr__(self):
        return f"InputExample(guid={self.guid}, instruction={self.instruction}, text={self.text}, " \
               f"prompt_examples={self.prompt_examples}, label={self.label})"


@dataclass
class InputEgDataset:
    train: List[InputExample]
    val: List[InputExample]
    test: List[InputExample]


def get_dataset_sizes(dataset: InputEgDataset):
    tr, vl, ts = dataset.train, dataset.val, dataset.test
    return dict(train_sz=len(tr), val_sz=len(vl), test_sz=len(ts))


class ListDataset(Dataset):
    def __init__(self, lst: List):
        self.lst = lst

    def __getitem__(self, idx):
        return self.lst[idx]

    def __len__(self):
        return len(self.lst)


def _load_dataset(dataset_name: str = None, leakage: bool = False) -> PersonalizedDataset:
    leak_str = 'leaked' if leakage else 'no_leak'
    data_dir_nm = os_join(dataset_name, f'user_data_{leak_str}.json')
    data_path = os_join(u.proj_path, u.dset_dir, data_dir_nm)

    d_log = dict(dataset_name=dataset_name, leakage=leakage, path=data_path)
    logger.info(f'Loading data w/ {pl.i(d_log)}...')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def _load_user_profiles(dataset_name: str = None) -> Dict:
    data_path = os_join(u.proj_path, 'user_context/profiles_n20', dataset_name, 'profiles.csv')
    data = {}
    with open(data_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['annotator_id']] = row['profile']
    return data

def load_dataset_with_prompts(
        dataset_name: str, leakage: bool = False,
        example_count: int = 1, max_example_count: int = 3, per_user: bool = True, seed: int = 42, use_user_profile: bool = False
) -> Union[InputEgDataset, Dict[str, InputEgDataset]]:
    """
    Process data for few-shot learning

    :param dataset_name: dataset name (e.g. goemotion)
        Will load a dataset of (user id => dataset split => sample id => sample data)
    :param leakage: See
    :param example_count: number of examples in few-shot prompt, per category
    :param max_example_count: maximum number of examples to use per category
    :param per_user: If True, return datasets for each user
    :param seed: random seed for sampling prompt examples
    """
    ret = dict()
    instruction = sconfig(f'datasets.{dataset_name}.instruction')
    dset = _load_dataset(dataset_name=dataset_name, leakage=leakage)
    user_profiles = _load_user_profiles(dataset_name=dataset_name) if use_user_profile else None

    for uid, dset_ in dset.items():
        def split2label_options(split: str) -> List[str]:
            return sorted(set().union(*[v['label'] for k, v in dset_[split].items()]))

        # Get all labels in the train split
        label_options = split2label_options('train')
        # not necessarily true, see `data2dataset_splits` when `leakage` is False
        # assert label_options == split2label_options('val') == split2label_options('test')

        lb2txts: Dict[str, List[str]] = {  # label => list of examples w/ that label
            label: [sample['text'] for id_, sample in dset_['train'].items() if label in sample['label']]
            for label in label_options
        }

        def get_split(split: str) -> List[InputExample]:
            """
            Add prompt example to each sample in the split
            """
            if seed:
                random.seed(seed)
            lst = []
            for sid, sample in dset_[split].items():
                text, label = sample['text'], sample['label']

                # take n random examples from each category for few-shot prompt
                prompt_egs = []
                for lb in label_options:
                    if split == 'train' and label == lb:  # filter out current sample if in train split
                        idx_curr_sample = lb2txts[lb].index(text)
                        txts = lb2txts[lb][:idx_curr_sample] + lb2txts[lb][idx_curr_sample+1:]
                    else:
                        txts = lb2txts[lb]
                    txts_selected = random.sample(txts, k=min(example_count, len(txts)))
                    prompt_egs += [(txt, lb) for txt in txts_selected]
                prompt_egs = prompt_egs[:max_example_count]  # keep only first `max_example_count` examples

                lst.append(
                    InputExample(guid=sid, instruction=instruction, text=text, prompt_examples=prompt_egs, label=label, user_profile=user_profiles[uid] if user_profiles else None)
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


_USER_IDS = Union[List[str], List[int]]
# For `EPIC`, user ids are hex chars
hex_pattern = re.compile(r'^[0-9a-f]+$')
# For `SubjectiveDiscourse`, user ids are like `worker_50`
sub_dis_pattern = re.compile(r'^worker_(?P<id>\d+)$')


def sort_user_ids(uids: _USER_IDS) -> _USER_IDS:
    if all(isinstance(uid, int) for uid in uids):
        return sorted(uids)
    else:
        assert all(isinstance(uid, str) for uid in uids)

        if all(uid.isdigit() for uid in uids):
            sort_fn = int
        elif all(hex_pattern.match(uid) is not None for uid in uids):
            def sort_fn(x):
                return int(x, 16)
        else:
            def sort_fn(x):
                match = sub_dis_pattern.match(x)
                assert match is not None
                return int(match.group('id'))
        return sorted(uids, key=sort_fn)


def iter_users(
        dataset: Dict[str, Any] = None, start_from: Union[str, int] = None, end_at: Union[str, int] = None,
        filter_fn: Callable = None
) -> List[str]:
    """
    Deterministic ordering of user ids

    :param dataset: dataset dict
    :param start_from: user id to start from, inclusive
    :param end_at: user id to end at, inclusive
    :param filter_fn: filter function to apply to user ids
    """
    ret = sort_user_ids(list(dataset.keys()))
    if start_from is not None:
        if isinstance(start_from, str):
            assert start_from.isdigit()
        else:
            start_from = str(start_from)
        idx_strt = ret.index(start_from)
        ret = ret[idx_strt:]  # remove ids before `start_from`
    if end_at is not None:
        if isinstance(end_at, str):
            assert end_at.isdigit()
        else:
            end_at = str(end_at)
        idx_end = ret.index(end_at)  # remove ids after `end_at`
        ret = ret[:idx_end+1]
    if filter_fn is not None:
        ret = [uid for uid in ret if filter_fn(uid)]
    return ret


if __name__ == '__main__':
    def check_data():
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
    # check_data()

    def check_add_template():
        dset = load_dataset_with_prompts(dataset_name='tweeteval')
        mic(dset.keys())
    check_add_template()

