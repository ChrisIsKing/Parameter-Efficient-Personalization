from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    return parser.parse_args()

if __name__ == '__main__':
    from stefutil import *
    
    args = parse_args()
    if args.output_dir is not None:
        output_dir = os_join(args.output_dir, 'gabhate')
        make_dirs(output_dir)

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'gabhate')
    data, headers = load_csv(os_join(dset_base_path, 'GabHateCorpus_annotations.tsv'), delimiter='\t', header=True)

    label_map = {0: 'Non-hateful', 1: 'Hateful'}
    user_data = defaultdict(dict)

    for row in data:
        post_id, user_id, text, label = row[0], row[1], row[2], label_map[int(row[3])]
        user_data[user_id][post_id] = dict(text=text, label=[label])
    save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
    mic(data2label_meta(data=user_data))
