import json
from os.path import join as os_join
from collections import defaultdict

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    from stefutil import *

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

    save_datasets(data=user_data, base_path=dset_base_path)
    mic(data2label_meta(data=user_data))
