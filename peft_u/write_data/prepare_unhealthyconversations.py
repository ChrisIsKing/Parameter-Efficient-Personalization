from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
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
        output_dir = os_join(args.output_dir, 'unhealthyconversations')
        make_dirs(output_dir)

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'unhealthyconversations')

    def run():
        data, headers = load_csv(os_join(dset_base_path, 'unhealthy_full.csv'), delimiter=',', header=True)

        label_map = {'1': 'healthy', '0': 'unhealthy'}
        user_data = defaultdict(dict)

        for row in tqdm(data, desc='Processing data'):
            post_id, text, user_id, label = row[0], row[1], row[3], row[9]
            if post_id not in user_data[user_id]:
                user_data[user_id][post_id] = dict(text=text, label=[label_map[label]])

        if args.num_samples is not None:
            keys = random.sample(list(user_data.keys()), args.num_samples)
            user_data = {k: user_data[k] for k in keys}

        save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
        mic(data2label_meta(data=user_data))
    run()

    def check_healthy_counts():
        import pandas as pd

        df = pd.read_csv(os_join(dset_base_path, 'unhealthy_full.csv'))
        mic(df['healthy'].value_counts())
    # check_healthy_counts()
