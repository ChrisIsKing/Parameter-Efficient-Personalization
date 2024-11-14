from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
from stefutil import *
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *

logger = get_logger('TweetEval Prep')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    return parser.parse_args()

if __name__ == '__main__':
    from stefutil import *

    args = parse_args()
    if args.output_dir is not None:
        output_dir = os_join(args.output_dir, 'tweeteval')
        make_dirs(output_dir)

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'tweeteval')

    def run():
        # Labels: [`Hateful`, `Non-hateful`]
        data, headers = load_csv(os_join(dset_base_path, 'annotations_g3.csv'), delimiter=",", header=True)
        uids = [headers[i] for i in range (2,len(headers))]
        user_data: PersonalizedData = defaultdict(dict)
        for row in data:
            post_id, text, user_labels = row[0], row[1], row[2:]
            for id_num, label in enumerate(user_labels):
                user_data[uids[id_num]][post_id] = dict(text=text, label=[label])

        save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
        mic(data2label_meta(data=user_data))
    run()

    def check_deterministic(fnm: str = None):
        import json

        path_old = f'{fnm}-old.json'
        with open(os_join(dset_base_path, path_old), 'r') as f:
            dset_ori = json.load(f)

        path_new = f'{fnm}.json'
        with open(os_join(dset_base_path, path_new), 'r') as f:
            dset_new = json.load(f)

        def get_uid2split_sizes(dset):
            return {uid: {split: len(samples) for split, samples in dset[uid].items()} for uid in dset}

        mic(get_uid2split_sizes(dset_ori))
        mic(get_uid2split_sizes(dset_new))
        assert dset_ori == dset_new
    # check_deterministic(fnm='user_data_no_leak')
    # check_deterministic(fnm='user_data_leaked')
