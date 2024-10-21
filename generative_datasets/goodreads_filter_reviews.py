import pandas as pd

min_num_user_answers = 20

reviews = pd.read_csv('generative_datasets/goodreads/goodreads_reviews_spoiler_raw.csv')

# filter to users with min number of reviews
reviews = reviews.groupby('user_id').filter(lambda x: len(x) >= min_num_user_answers)
# filter down to reviews on books with descriptions
books = pd.read_csv('generative_datasets/goodreads/goodreads_books.csv')
reviews = reviews[reviews['book_id'].isin(books['Id'])]
# remove structured spoiler text from reviews
# reviews = reviews['review_text'].str.replace("(view spoiler)[ ","").str.replace(" (hide spoiler)]","")

print(f"Writing {len(reviews)} lines to csv")
reviews.to_csv('generative_datasets/goodreads/goodreads_reviews_filtered.csv')