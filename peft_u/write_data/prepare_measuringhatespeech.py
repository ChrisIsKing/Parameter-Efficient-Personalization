import os
from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
import datasets
from tqdm import trange
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    parser.add_argument('--num_samples', default=None, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    from stefutil import *

    args = parse_args()
    if args.output_dir is not None:
        output_dir = os_join(args.output_dir, 'measuringhatespeech')
        make_dirs(output_dir)

    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
    df = dataset['train'].to_pandas()

    user_data = defaultdict(dict)

    for i in trange(len(df), desc='Converting data'):
        post_id, user_id = str(df['comment_id'][i]), str(df['annotator_id'][i])
        text, label = df['text'][i], str(int(df['hatespeech'][i]))
        user_data[user_id][post_id] = dict(text=text, label=[label])

    if args.num_samples is not None:
        keys = random.sample(list(user_data.keys()), args.num_samples)
        user_data = {k: user_data[k] for k in keys}

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'measuringhatespeech')
    os.makedirs(dset_base_path, exist_ok=True)
    save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
    mic(data2label_meta(data=user_data))
