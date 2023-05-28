import ast
from os.path import join as os_join
from collections import defaultdict

from tqdm import tqdm

from stefutil import *
from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


logger = get_logger('Subjective Discourse Write')


if __name__ == '__main__':
    dset_base_path = os_join(u.proj_path, u.dset_dir, 'subjectivediscourse')

    data_path = os_join(dset_base_path, 'with_features_annotated_questions_responses_gold.tsv')
    data, headers = load_csv(data_path, delimiter='\t', header=True)

    user_data = defaultdict(dict)

    label_set, sentiment_set = set(), set()

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
            user_data[user][post_id] = dict(
                text=text, label=[labels[i]], q_sentiment=[q_sentiments[i]], r_sentiment=[r_sentiments[i]]
            )

    save_datasets(data=user_data, base_path=dset_base_path)
    logger.info(pl.i({'Label Set': sorted(label_set), 'Sentiment Set': sorted(sentiment_set)}))
