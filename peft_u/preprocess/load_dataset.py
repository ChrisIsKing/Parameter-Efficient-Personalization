import json
import random
from os.path import join as os_join

from dataclasses import dataclass
from typing import List, Dict, Union

from torch.utils.data import Dataset

from stefutil import *
from peft_u.util import *
from peft_u.preprocess.convert_data_format import PersonalizedDataset


__all__ = [
    'InputEgDataset', 'ListDataset',
    'load_dataset_with_prompts'
]


logger = get_logger('Load Dataset')


class InputExample:
    def __init__(self, guid: Union[int, str], instruction: str = None, text: str = None, prompt_examples: List = None,
                 label: List[str] = None) -> None:
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


def load_dataset_with_prompts(
        dataset_name: str, leakage: bool = False,
        example_count: int = 1, max_example_count: int = 3, per_user: bool = True
) -> Union[InputEgDataset, Dict[str, InputEgDataset]]:
    """
    Process data for few-shot learning

    :param dataset_name: dataset name (e.g. goemotion)
        Will load a dataset of (user id => dataset split => sample id => sample data)
    :param leakage: See
    :param example_count: number of examples in few-shot prompt, per category
    :param max_example_count: maximum number of examples to use per category
    :param per_user: If True, return datasets for each user
    """
    ret = dict()
    instruction = sconfig(f'datasets.{dataset_name}.instruction')
    dset = _load_dataset(dataset_name=dataset_name, leakage=leakage)

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
            lst = []
            for sid, sample in dset_[split].items():
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
