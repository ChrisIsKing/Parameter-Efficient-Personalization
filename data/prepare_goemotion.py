import json
from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from utils import *
from peft_u.util import *


if __name__ == '__main__':
    dset_base_path = os_join(u.proj_path, u.dset_dir, 'goemotion')

    with open(os_join(dset_base_path, 'ekman_mapping.json')) as f:
        label_map = json.load(f)
    label_map = flip_dict_of_lists(label_map)
    label_map['neutral'] = 'neutral'

    csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv',]

    user_data = defaultdict(dict)
    for csv_file in tqdm(csv_files, desc='Processing CSVs'):
        data, header = load_csv(os_join(dset_base_path, csv_file), delimiter=",", header=True)
        for row in data:
            text, post_id, user_id = row[0], row[1], row[7]
            label = list(set([label_map[header[i+9]] for i, item in enumerate(row[9:]) if item == '1']))
            if 'neutral' in label or len(label) < 1:
                continue

            if post_id not in user_data[user_id]:  # Sample not already added
                user_data[user_id][post_id] = dict(text=text, label=label)

    save_datasets(data=user_data, base_path=dset_base_path)
