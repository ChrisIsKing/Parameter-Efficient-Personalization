from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    dset_base_path = os_join(u.proj_path, u.dset_dir, 'unhealthyconversations')

    data, headers = load_csv(os_join(dset_base_path, 'unhealthy_full.csv'), delimiter=',', header=True)

    user_data = defaultdict(dict)

    for row in tqdm(data, desc='Processing data'):
        post_id, text, user_id, label = row[0], row[1], row[3], row[9]
        if post_id not in user_data[user_id]:
            user_data[user_id][post_id] = dict(text=text, label=[label])
    save_datasets(data=user_data, base_path=dset_base_path)
