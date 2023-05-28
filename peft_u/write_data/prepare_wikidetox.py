from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
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
    save_datasets(data=user_data, base_path=dset_base_path)
