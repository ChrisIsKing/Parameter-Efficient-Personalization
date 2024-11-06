import json
from os.path import join as os_join
from collections import Counter, defaultdict
from argparse import ArgumentParser
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='Path to output directory')
    return parser.parse_args()

def _most_common(lst, tie_breaker: str = None):
    """
    Returns the most common element in a list.
    If the count of the top 2 elements are the same then `tie_breaker` is returned.
    """
    c = Counter(lst)
    top_2 = c.most_common(2)
    if len(top_2) > 1:
        if top_2[0][1] == top_2[1][1]:
            return tie_breaker
    return top_2[0][0]


def prepare_data(data, test_ids=None, random_state=42):
    """
    Prepare hatexplain data.
    """
    set_seed(random_state)
    ruleset, normal_examples, global_examples = defaultdict(dict), dict(), defaultdict(dict)

    # for k, v in data.items():
    for k in sorted(data.keys()):
        v = data[k]

        label = _most_common(lst=[annotator['label'] for annotator in v['annotators']], tie_breaker='undecided')
        group = [
            item for item, count in
            Counter([group for annotator in v['annotators'] for group in annotator['target']]).items() if count > 1
        ]
        if not group:
            group = ['None']
        post_id, post_text = v['post_id'], ' '.join(v['post_tokens'])
        if label == 'normal':
            v['text'], v['majority_label'] = post_text, ['normal']
            v['majority_group'] = group
            normal_examples[k] = v

            v['rationale_spans'] = []
            if test_ids:
                split = None
                if post_id in test_ids['train']:
                    split = 'train'
                elif post_id in test_ids['test']:
                    split = 'test'
                elif post_id in test_ids['val']:
                    split = 'val'
                global_examples[split][k] = v
        elif label != "undecided":
            global_examples['all'][k] = v | dict(
                majority_label=[label],
                text=post_text,
                rationale_spans=[],
                majority_group=group
            )

        for index, annotator in enumerate(v['annotators']):
            annot_id = annotator['annotator_id']
            if v['rationales']:
                d = dict(
                    post_tokens=v['post_tokens'],
                    text=post_text,
                    label=[v['annotators'][index]['label']],
                    target_group=v['annotators'][index]['target'],
                    rationale_spans=[],
                    majority_label=[label],
                    majority_group=group
                )

                if len(v['rationales']) <= index:
                    ruleset[annot_id][post_id] = d | dict(
                        rationale=[],
                        rationale_tokens=[]
                    )
                else:
                    d |= dict(
                        rationale=v['rationales'][index],
                        rationale_tokens=[
                            token for i, token in enumerate(v['post_tokens']) if v['rationales'][index][i] == 1
                        ]
                    )
                    queue = []
                    for flag, value in zip(v['rationales'][index], v['post_tokens']):
                        if flag:
                            queue.append(value)
                        elif queue:
                            rationale = ' '.join(queue)
                            d['rationale_spans'].append(rationale)
                            global_examples['all'][k]['rationale_spans'].append(rationale)
                            queue = []
                    if queue:
                        rationale = ' '.join(queue)
                        d['rationale_spans'].append(rationale)
                        global_examples['all'][k]['rationale_spans'].append(rationale)
                    ruleset[annot_id][post_id] = d
            else:
                ruleset[annot_id][post_id] = dict(
                    post_tokens=v['post_tokens'],
                    text=post_text,
                    label=[v['annotators'][index]['label']],
                    target_group=v['annotators'][index]['target'],
                    rationale_spans=[],
                    majority_label=[label],
                    majority_group=group
                )
    return ruleset, normal_examples, global_examples


if __name__ == '__main__':
    from stefutil import *

    def run():
        args = parse_args()
        if args.output_dir is not None:
            output_dir = os_join(args.output_dir, 'hatexplain')
            make_dirs(output_dir)

        # Labels: [`normal`, `hatespeech`, `offensive`]
        dset_base_path = os_join(u.proj_path, u.dset_dir, 'hatexplain')

        with open(os_join(dset_base_path, 'dataset.json')) as fl:
            data = json.load(fl)
        with open(os_join(dset_base_path, 'post_id_divisions.json')) as fl:
            test_ids = json.load(fl)

        user_data, normal_examples, global_examples = prepare_data(data, test_ids)

        save_datasets(data=user_data, base_path=output_dir if args.output_dir is not None else dset_base_path)
        mic(data2label_meta(data=user_data))
    run()
