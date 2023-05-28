import os
from os.path import join as os_join
from collections import defaultdict

import datasets
from tqdm import trange

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    from stefutil import *

    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
    df = dataset['train'].to_pandas()

    user_data = defaultdict(dict)

    for i in trange(len(df), desc='Converting data'):
        post_id, user_id = str(df['comment_id'][i]), str(df['annotator_id'][i])
        text, label = df['text'][i], str(int(df['hatespeech'][i]))
        user_data[user_id][post_id] = dict(text=text, label=[label])

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'measuringhatespeech')
    os.makedirs(dset_base_path, exist_ok=True)
    save_datasets(data=user_data, base_path=dset_base_path)
    mic(data2label_meta(data=user_data))
