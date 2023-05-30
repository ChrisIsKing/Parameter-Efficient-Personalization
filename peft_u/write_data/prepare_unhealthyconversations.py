from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    from stefutil import *

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'unhealthyconversations')
    data, headers = load_csv(os_join(dset_base_path, 'unhealthy_full.csv'), delimiter=',', header=True)

    label_map = {'1': 'healthy', '0': 'unhealthy'}
    user_data = defaultdict(dict)

    for row in tqdm(data, desc='Processing data'):
        post_id, text, user_id, label = row[0], row[1], row[3], row[9]
        if post_id not in user_data[user_id]:
            user_data[user_id][post_id] = dict(text=text, label=[label_map[label]])
    save_datasets(data=user_data, base_path=dset_base_path)
    mic(data2label_meta(data=user_data))

    def check_healthy_counts():
        import pandas as pd

        df = pd.read_csv(os_join(dset_base_path, 'unhealthy_full.csv'))
        mic(df['healthy'].value_counts())
    # check_healthy_counts()
