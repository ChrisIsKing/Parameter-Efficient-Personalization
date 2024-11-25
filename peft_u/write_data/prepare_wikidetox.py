from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    from stefutil import *

    args = parse_args()
    if args.output_dir is not None:
        output_dir = os_join(args.output_dir, 'wikidetox')
        make_dirs(output_dir)

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'wikidetox')

    annot_path = os_join(dset_base_path, 'aggression_annotations.tsv')
    annotation_data, annotation_headers = load_csv(annot_path, delimiter="\t", header=True)
    txt_path = os_join(dset_base_path, 'aggression_annotated_comments.tsv')
    text_data, text_headers = load_csv(txt_path, delimiter="\t", header=True)

    label_map = {0: 'Normal', 1: 'Aggressive'}

    id2text = dict()
    for i, row in enumerate(tqdm(text_data, desc='Building id => text map')):
        post_id, text = row[0], row[1]
        id2text[post_id] = text

    user_data = defaultdict(dict)

    for i, row in enumerate(tqdm(annotation_data, desc='Processing data')):
        post_id, user_id, label = row[0], row[1], row[2]
        if post_id not in user_data[user_id]:
            user_data[user_id][post_id] = dict(text=id2text[post_id], label=[label_map[int(float(label))]])
    
    if args.num_samples is not None:
        random.seed(args.seed)
        keys = random.sample(list(user_data.keys()), args.num_samples)
        user_data = {k: user_data[k] for k in keys}

    save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
    mic(data2label_meta(data=user_data))
