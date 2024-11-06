import os
import pandas as pd

min_num_user_answers = 20

root_dir = 'generative_datasets/goodreads/books/'

print("Getting reviews")
reviews = pd.read_csv('generative_datasets/goodreads/goodreads_reviews_spoiler_raw.csv')
reviews = reviews.groupby('user_id').filter(lambda x: len(x) >= min_num_user_answers)
print(reviews)

books_filtered = pd.DataFrame()
concat_list = []
# look through each list of books, find books with reviews
for directory, subdirectories, files in os.walk(root_dir):
    for file in files:
        print(f'Parsing {file}')
        books = pd.read_csv(root_dir+file)
        books_needed = books[books['Id'].isin(reviews['book_id'])]
        concat_list.append(books_needed)
books_filtered = pd.concat(concat_list)
# remove blank descriptions
books_filtered = books_filtered[(books_filtered['Description']!='')&(books_filtered['Description'].notna())]
# concatenate other book data -- No need, no new info
# book_work_data = pd.read_csv('generative_datasets/goodreads/goodreads_book_works.csv')[['best_book_id','books_count','default_description_language_code','reviews_cou']]
# books_filtered = pd.merge(books_filtered,book_work_data,how='left',left_on='Id',right_on='best_book_id').drop('best_book_id',axis=1)
print("Writing to csv")
books_filtered.to_csv('generative_datasets/goodreads/goodreads_books.csv')