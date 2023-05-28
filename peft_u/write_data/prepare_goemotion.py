import json
from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    dset_base_path = os_join(u.proj_path, u.dset_dir, 'goemotion')

    with open(os_join(dset_base_path, 'ekman_mapping.json')) as f:
        e2detailed_e = json.load(f)

    detailed_e2e = {detailed_e: e for e, detailed_es in e2detailed_e.items() for detailed_e in detailed_es}
    detailed_e2e['neutral'] = 'neutral'

    csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']

    user_data = defaultdict(dict)
    for csv_file in tqdm(csv_files, desc='Processing CSVs'):
        data, header = load_csv(os_join(dset_base_path, csv_file), delimiter=",", header=True)
        for row in data:
            text, post_id, user_id = row[0], row[1], row[7]
            label = list(set([detailed_e2e[header[i + 9]] for i, item in enumerate(row[9:]) if item == '1']))
            if 'neutral' in label or len(label) < 1:
                continue

            if post_id not in user_data[user_id]:  # Sample not already added
                user_data[user_id][post_id] = dict(text=text, label=label)

    save_datasets(data=user_data, base_path=dset_base_path)
