import os
from os.path import join as os_join
import fnmatch
from collections import defaultdict
from argparse import ArgumentParser
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--substack', type=str)
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    parser.add_argument('--num_samples', default=None, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    from stefutil import *

    args = parse_args()
    substack = args.substack

    dset_base_path = os_join(u.proj_path, u.dset_dir, substack)#f'{substack}.stackexchange.com')

    csv_files = [f for f in os.listdir(dset_base_path) if fnmatch.fnmatch(f, f"{substack}*.csv")]
    data, headers = load_csv(os_join(dset_base_path, csv_files[0]), delimiter=',', header=True)
    if len(csv_files) > 1:
        for csv_file in csv_files[1:]:
            next_data, headers = load_csv(os_join(dset_base_path, csv_file), delimiter=',', header=True)
            data += next_data

    # label_map = {0: 'Non-hateful', 1: 'Hateful'}
    user_data = defaultdict(dict)

    for row in data:
        index, user_id, answer, title, tags, question, question_id = int(float(row[0])), int(float(row[1])), row[2], row[3], row[4], row[5], int(float(row[6]))
        user_data[user_id][question_id] = dict(question=f"{title} {tags} {question}", answer=[answer])
    # from itertools import islice
    # user_data = dict(islice(user_data.items(), 5))
    if args.num_samples is not None:
        keys = random.sample(list(user_data.keys()), args.num_samples)
        user_data = {k: user_data[k] for k in keys}

    save_datasets(data=user_data, base_path=dset_base_path, is_generative=True, label_key='answer')
    # mic(data2label_meta(data=user_data))
