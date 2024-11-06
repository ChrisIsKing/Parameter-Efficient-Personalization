import json
from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    return parser.parse_args()

if __name__ == '__main__':
    from stefutil import *

    args = parse_args()
    if args.output_dir is not None:
        output_dir = os_join(args.output_dir, 'cockamamie')
        make_dirs(output_dir)

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'cockamamie')
    with open(os_join(dset_base_path, 'cockamamie.json')) as fl:
        data = json.load(fl)

    user_data, post_map = defaultdict(dict), dict()

    for item in data['word_ratings']['votes']:
        for key, value in item.items():
            # Loop through yes and no votes
            for voter in value['no_votes']:
                if key not in post_map:
                    post_map[key] = len(post_map)
                user_data[voter][post_map[key]] = dict(text=key, label=["no"])
            for voter in value['yes_votes']:
                if key not in post_map:
                    post_map[key] = len(post_map)
                user_data[voter][post_map[key]] = dict(text=key, label=["yes"])

    save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
    mic(data2label_meta(data=user_data))
