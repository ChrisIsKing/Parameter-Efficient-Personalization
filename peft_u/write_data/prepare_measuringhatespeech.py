import os
from os.path import join as os_join

import datasets

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
    df = dataset['train'].to_pandas()

    user_data = {}

    for i in range(len(df)):
        post_id = str(df['comment_id'][i])
        user_id = str(df['annotator_id'][i])
        text = df['text'][i]
        label = str(int(df['hatespeech'][i]))

        if user_id not in user_data:
            user_data[user_id] = {}

        user_data[user_id][post_id] = {'text': text, 'label': [label]}

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'measuringhatespeech')
    os.makedirs(dset_base_path, exist_ok=True)
    save_datasets(data=user_data, base_path=dset_base_path)
