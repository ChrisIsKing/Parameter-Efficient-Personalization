import pandas as pd

file_in = 'generative_datasets/goodreads/goodreads_book_works.json'
file_out = 'generative_datasets/goodreads/goodreads_reviews_spoiler_raw.csv'

chunksize = 100000
df_chunks = pd.read_json(file_in, lines=True, chunksize=chunksize)
df = pd.concat(df_chunks, ignore_index=True)
df.to_csv(file_out)