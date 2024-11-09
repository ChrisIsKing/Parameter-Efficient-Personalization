from os.path import join as os_join
from collections import defaultdict

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *

import sys


if __name__ == '__main__':
    from stefutil import *

    substack = sys.argv[1]

    dset_base_path = os_join(u.proj_path, u.dset_dir, f'{substack}.stackexchange.com')
    data, headers = load_csv(os_join(dset_base_path, f'{substack}.csv'), delimiter=',', header=True)

    # label_map = {0: 'Non-hateful', 1: 'Hateful'}
    user_data = defaultdict(dict)

    for row in data:
        index, user_id, answer, title, tags, question, question_id = int(float(row[0])), int(float(row[1])), row[2], row[3], row[4], row[5], int(float(row[6]))
        user_data[user_id][question_id] = dict(question=f"{title} {tags} {question}", answer=[answer])
    # from itertools import islice
    # user_data = dict(islice(user_data.items(), 5))
    save_datasets(data=user_data, base_path=dset_base_path, is_generative=True, label_key='answer')
    # mic(data2label_meta(data=user_data))
