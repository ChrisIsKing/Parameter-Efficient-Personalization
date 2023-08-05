from os.path import join as os_join
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from stefutil import *
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


logger = get_logger('Prep EPIC')


if __name__ == '__main__':
    mic.output_width = 256

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'epic')

    df = pd.read_csv(os_join(dset_base_path, 'EPICorpus.csv'))
    # mic(df)

    user_data = defaultdict(dict)
    label_map = {'iro': 'Ironic', 'not': 'Non-ironic'}  # TODO: what should be the textual labels called?

    d_log = {'#annotators': len(df['user'].unique()), '#samples': len(df['id_original'].unique())}
    logger.info(pl.i(d_log))
    for i, row in tqdm(df.iterrows(), desc='Converting data', total=len(df)):
        post_id, user_id = row.id_original, row.user
        # TODO: each sample is a text-pair, how to handle?
        text = f'message: "{row.parent_text}"\n\nreply: "{row.text}"'
        label = label_map[row.label]
        user_data[user_id][post_id] = dict(text=text, label=[label])
        # mic(row)
        # mic(user_data)
        # raise NotImplementedError
    save_datasets(data=user_data, base_path=dset_base_path)
    mic(data2label_meta(data=user_data))
