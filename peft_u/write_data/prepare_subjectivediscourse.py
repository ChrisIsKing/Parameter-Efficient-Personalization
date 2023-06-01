import os
import ast
from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from stefutil import *
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


logger = get_logger('Subjective Discourse Write')


if __name__ == '__main__':
    def run(label_key: str = 'response'):
        ca.check_mismatch('Classification type', label_key, ['response', 'question_sentiment', 'response_sentiment'])

        dnm = 'subjectivediscourse'
        data_path = os_join(u.proj_path, u.dset_dir, dnm, 'with_features_annotated_questions_responses_gold.tsv')
        data, headers = load_csv(data_path, delimiter='\t', header=True)

        user_data = defaultdict(dict)
        label_set, sentiment_set = set(), set()
        _label_map = dict(
            somewhatNegative='somewhat-negative',
            somewhatPositive='somewhat-positive',
            veryNegative='very-negative',
            veryPositive='very-positive'
        )

        def label_map(x: str) -> str:
            return _label_map.get(x, x)

        for row in tqdm(data, desc='Processing data'):
            post_id = row[3]
            users = ast.literal_eval(row[17])

            labels = ast.literal_eval(row[11])
            q_sentiments = ast.literal_eval(row[21])
            r_sentiments = ast.literal_eval(row[18])
            text = '%s \n %s \n %s \n %s' % (row[5], row[6], row[7], row[8])

            for label in labels:
                label_set.add(label)
            for sentiment in q_sentiments:
                sentiment_set.add(sentiment)
            for sentiment in r_sentiments:
                sentiment_set.add(sentiment)

            for i, user in enumerate(users):
                if label_key == 'response':
                    label = [labels[i]]
                elif label_key == 'question_sentiment':
                    label = [label_map(q_sentiments[i])]
                else:
                    assert label_key == 'response_sentiment'
                    label = [label_map(r_sentiments[i])]

                user_data[user][post_id] = dict(text=text, label=label)
        dset_out_path = os_join(u.proj_path, u.dset_dir, f'{dnm}_{label_key}')
        os.makedirs(dset_out_path, exist_ok=True)
        save_datasets(data=user_data, base_path=dset_out_path)
        logger.info(pl.i({'Label Set': sorted(label_set), 'Sentiment Set': sorted(sentiment_set)}))
        mic(data2label_meta(data=user_data))
    run(label_key='response')
    run(label_key='question_sentiment')
    run(label_key='response_sentiment')
