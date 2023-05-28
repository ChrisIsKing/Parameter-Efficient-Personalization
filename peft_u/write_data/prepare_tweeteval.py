from os.path import join as os_join
from collections import defaultdict

from stefutil import *
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


logger = get_logger('TweetEval Prep')


if __name__ == '__main__':
    from stefutil import *

    # Labels: [`Hateful`, `Non-hateful`]
    dset_base_path = os_join(u.proj_path, u.dset_dir, 'tweeteval')
    data, headers = load_csv(os_join(dset_base_path, 'annotations_g3.csv'), delimiter=",", header=True)

    user_data: PersonalizedData = defaultdict(dict)
    for row in data:
        post_id, text, user_labels = row[0], row[1], row[2:]
        for uid, label in enumerate(user_labels):
            user_data[uid][post_id] = dict(text=text, label=[label])

    save_datasets(data=user_data, base_path=dset_base_path)
    mic(data2label_meta(data=user_data))
