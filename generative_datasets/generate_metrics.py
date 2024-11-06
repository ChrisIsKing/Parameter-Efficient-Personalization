import pandas as pd
import os

min_num_user_answers = 20

metrics = pd.DataFrame(columns=["Name", "# Users", "Avg # examples per user", "Avg # users per text", "Krippendorff’s Alpha"]).set_index("Name")

datasets = [f for f in os.listdir('generative_datasets') if os.path.isdir(os.path.join('generative_datasets', f)) and f.endswith('.stackexchange.com')]

for dataset in datasets:
    questions = pd.read_xml(f'generative_datasets/{dataset}/Questions.xml')
    answers = pd.read_xml(f'generative_datasets/{dataset}/Answers.xml')

    num_users = len(answers['OwnerUserId'].unique())
    avg_examples = answers.groupby('OwnerUserId')['Body'].count().mean()
    avg_users = questions.groupby('Id')['AnswerCount'].first().mean()

    metrics.loc[dataset, metrics.columns] = [num_users, avg_examples, avg_users, None]

books = pd.read_csv('generative_datasets/goodreads/goodreads_books.csv')
reviews = pd.read_csv('generative_datasets/goodreads/goodreads_reviews_filtered.csv')

num_users = len(reviews['user_id'].unique())
avg_examples = reviews.groupby('user_id')['review_text'].count().mean()
avg_users = books.groupby('Id')['Count of text reviews'].first().mean()
metrics.loc['goodreads', metrics.columns] = [num_users, avg_examples, avg_users, None]

print(metrics)
# print(f"Writing {len(reviews)} lines to csv")
metrics.to_csv('generative_datasets/dataset_metrics.csv')