import os
from os.path import join as os_join
import fnmatch
from collections import defaultdict

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *

import sys


if __name__ == '__main__':
    from stefutil import *

    substack = sys.argv[1]

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
    save_datasets(data=user_data, base_path=dset_base_path, is_generative=True, label_key='answer')
    # mic(data2label_meta(data=user_data))
