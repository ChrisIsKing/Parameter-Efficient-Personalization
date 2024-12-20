from os.path import join as os_join
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
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
        output_dir = os_join(args.output_dir, 'studemo')
        make_dirs(output_dir)

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'studemo')

    annot_path = os_join(dset_base_path, 'annotation_data.csv')
    annotation_data, annotation_headers = load_csv(annot_path, delimiter=",", header=True)
    text_data, text_headers = load_csv(os_join(dset_base_path, 'text_data.csv'), delimiter=",", header=True)

    id2text = {}
    for i, row in enumerate(tqdm(text_data, desc='Building id => text map')):
        post_id, text = row[0], row[1]
        id2text[post_id] = text

    user_data = defaultdict(dict)

    for i, row in enumerate(tqdm(annotation_data, desc='Processing data')):
        post_id, user_id, labels = row[0], row[1], row[2:]
        if post_id not in user_data[user_id]:
            label = [annotation_headers[i+2] for i, l in enumerate(labels) if float(l) >= 1]
            if label:
                user_data[user_id][post_id] = dict(text=id2text[post_id], label=label)

    save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
    mic(data2label_meta(data=user_data))
