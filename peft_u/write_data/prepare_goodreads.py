import os
from os.path import join as os_join
import fnmatch
from collections import defaultdict

from peft_u.util import *
from peft_u.preprocess.convert_data_format import *


if __name__ == '__main__':
    from stefutil import *

    dset_base_path = os_join(u.proj_path, u.dset_dir, 'goodreads')

    csv_files = [f for f in os.listdir(dset_base_path) if fnmatch.fnmatch(f, f"goodreads*.csv")]
    data, headers = load_csv(os_join(dset_base_path, csv_files[0]), delimiter=',', header=True)
    if len(csv_files) > 1:
        for csv_file in csv_files[1:]:
            next_data, headers = load_csv(os_join(dset_base_path, csv_file), delimiter=',', header=True)
            data += next_data

    # label_map = {0: 'Non-hateful', 1: 'Hateful'}
    user_data = defaultdict(dict)

    print(data[0][19])
    for row in data:
    #     Index(['Unnamed: 0', 'Id', 'Name', 'Authors', 'ISBN', 'Rating', 'PublishYear',
    #    'PublishMonth', 'PublishDay', 'Publisher', 'RatingDist5', 'RatingDist4',
    #    'RatingDist3', 'RatingDist2', 'RatingDist1', 'RatingDistTotal',
    #    'CountsOfReview', 'Language', 'PagesNumber', 'Description',
    #    'pagesNumber', 'Count of text reviews', 'user_id', 'book_id',
    #    'review_id', 'rating', 'review_text', 'date_added', 'date_updated',
    #    'read_at', 'started_at', 'n_votes', 'n_comments'],
        book_id, user_id, book_description, review, title, authors = row[23], row[22], row[19], row[26], row[2], row[3]
        user_data[user_id][book_id] = dict(book_description=f'"{title}" by {authors}: {book_description}', review=[review])
    save_datasets(data=user_data, base_path=dset_base_path, is_generative=True, label_key='review')
    # mic(data2label_meta(data=user_data))
