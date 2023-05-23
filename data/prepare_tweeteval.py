import json
import nltk
from os.path import join as os_join
from nltk.metrics.distance import masi_distance

from stefutil import *
from utils import *
from peft_u.util import *


logger = get_logger('TweetEval Prep')


if __name__ == '__main__':
    dset_base_path = os_join(u.proj_path, u.dset_dir, 'tweeteval')
    data, headers = load_csv(os_join(dset_base_path, 'annotations_g3.csv'), delimiter=",", header=True)

    label_map = {"Hateful": 1, "Non-hateful": 0}

    user_data = {}

    for row in data:
        post_id = row[0]
        text = row[1]
        user_labels = row[2:]
        for i, label in enumerate(user_labels):
            if i not in user_data:
                user_data[i] = {}
            user_data[i][post_id] = {"text": text, "label": [label]}

    num_annotators = len(user_data)
    num_examples = sum([len(v) for k, v in user_data.items()])

    user_data_leaked, agreement_data = split_data(user_data, 0.8, random_state=42, leakage=True)

    user_data_no_leak, agreement_data_ = split_data(user_data, 0.8, random_state=42, leakage=False)
    assert set(agreement_data) == set(agreement_data_)  # sanity check

    masi_task = nltk.AnnotationTask(distance=masi_distance)
    masi_task.load_array(agreement_data)
    d_log = {
        'Krippendorff\'s alpha': masi_task.alpha(),
        '#Users': num_annotators,
        '#Examples': num_examples,
        'Avg #Examples/User': num_examples/num_annotators,
        'Avg #Users/Example': avg_num_users_per_example(user_data),
    }
    logger.info(pl.i(d_log))

    # Save the data
    fnm_leaked = os_join(dset_base_path, 'user_data_leaked.json')
    with open(fnm_leaked, 'w') as f:
        json.dump(user_data_leaked, f)
    logger.info(f'Leaked data saved to {pl.i(fnm_leaked)}')

    fnm_no_leak = os_join(dset_base_path, 'user_data_no_leak.json')
    with open(fnm_no_leak, 'w') as f:
        json.dump(user_data_no_leak, f)
    logger.info(f'Non-leaked data saved to {pl.i(fnm_no_leak)}')
