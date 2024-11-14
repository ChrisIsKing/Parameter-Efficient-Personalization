from openai import OpenAI
client = OpenAI()

import os
import csv
import pandas as pd
import itertools
import json
import glob
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from user_context.load_cockamamie import load_cockamamie
from user_context.load_epic import load_epic
from user_context.load_gabhate import load_gabhate
from user_context.load_goemotion import load_goemotion
from user_context.load_hatexplain import load_hatexplain
from user_context.load_studemo import load_studemo
from user_context.load_subjectivediscourse import load_subjectivediscourse
from user_context.load_tweeteval import load_tweeteval
from user_context.load_unhealthyconversations import load_unhealthyconversations
from user_context.load_wikidetox import load_wikidetox

classification_prompt = """
Attached are sub-samples of the {dataset} dataset for {batch_size} annotators (identified by the column 'annotator_id'). The dataset consists of labeled classification data. A description of the dataset is provided below:

Dataset description:
{data_description}

Perform the following steps and return a annotator profile ({{profile}}).

(1) Determine the schema of the dataset:
    (1.a) The columns '{user_id_col}' and '{text_id_col}' are the IDs for the annotator and the data to be labeled, respectively
    (1.b) Identify which column(s) contain the label(s) which classify the content.
    (1.c) Determine and describe the classification task and the meanings of the labels for the annotators based on the input data and the labels. For example, if the classification labels include 'happy', 'sad', 'angry', 'excited' and the input data are tweets, then this may be a sentiment analysis task to analyze the sentiment of a tweet.
(2) For each annotator, build an annotator profile:
    (2.a) Gather K clusters of examples annotated by the user which are similar in annotation perspective
    (2.b) For each cluster 1...K:
        (2.b.i) Generate a written synopsis that describes the annotator's perspective in labeling those examples. Notably, this is not the tone or perspective of the text itself, but rather what the annotator's interpretation of the text says about them and their perspective.
    (2.c) Given the K synopses, write a summary of the annotator's perspective in performing the labeling task
    (2.d) Given the summary, make some assumptions about the annotator and construct a written annotator profile ({{profile}}) as described below.

Annotator profile:
The annotator profile should be written as a bio, describing the annotator's biases, preferences, opinions, and any other deductions that can be made about their personality/profile. Do not mention any variable names or other annotators (or their ordering), or discuss your though process in the profile. Be concise and focus on the aspects which distinguish the annotator's perspective from the norm.

Combine the results into a json dictionary with 'annotator_id' as the key and the annotator profile as the value.
"""

generative_prompt = """
Attached are sub-samples of the {dataset} dataset for {batch_size} user (identified by the column '{user_id_col}'). The dataset consists of generative response data based on a prompt (ex. Q&A). A description of the dataset is provided below:

Dataset description:
{data_description}

Perform the following steps and return a user profile ({{profile}}).

(1) Determine the schema of the dataset:
    (1.a) The column '{user_id_col}' and '{text_id_col}' are the IDs for the user responding to the prompt and the prompt itself, respectively
    (1.b) Identify which column(s) contain the response(s) to the prompt.
(2) For each answering user, build a user profile:
    (2.a) Gather K clusters of examples answered by the user which are similar in perspective
    (2.b) For each cluster 1...K:
        (2.b.i) Generate a written synopsis that describes the user's perspective in responding to those prompts. Notably, this is not the tone or perspective of the prompt itself, but rather what the user's response to the prompt says about them and their perspective.
    (2.c) Given the K synopses, write a summary of the user's perspective in performing the response task
    (2.d) Given the summary, make some assumptions about the user and construct a written user profile ({{profile}}) as described below.

User profile:
The user profile should be written as a bio, describing the user's biases, preferences, opinions, and any other deductions that can be made about their personality/profile. Do not mention any variable names or other users (or their ordering), or discuss your thought process in the profile. Be concise and focus on the aspects which distinguish the user's perspective from the norm.

Combine the results into a json dictionary with '{user_id_col}' as the key and the user profile as the value.
"""

# Function to split groupby object into batches of {batch_size} groups
def split_into_batches(grouped, batch_size):
    # Convert the groupby object to an iterator
    iterator = iter(grouped)
    
    # Create a list to hold all the batches
    batches = []
    
    while True:
        # Take the next batch of groups
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        # Concatenate the DataFrame parts corresponding to this batch of groups
        batches.append(pd.concat([group for _, group in batch]))
    
    return batches

# Function to read a file and return its content
def read_file(file_path, N=1000):
    ret = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        for i in range(N):
            line = file.readline()
            ret += line
        return ret

# num_groups = 100
num_samples = 20
batch_size = 20

datasets = [f for f in os.listdir('data/')]
parent_folder = f'user_context/profiles_n{batch_size}/'

for dataset in datasets:
    logger.info(f'{dataset}:')
    # read content from a file
    filenames = glob.glob(f'data/{dataset}/{dataset}*.csv')
    # set directories
    profile_file = f'{parent_folder}/{dataset}/profiles.csv'
    log_file = f'{parent_folder}/{dataset}/logs.csv'

    # create files if don't exist
    if not os.path.exists(f'{parent_folder}/'):
        os.makedirs(f'{parent_folder}/')
    if not os.path.exists(f'{parent_folder}/{dataset}'):
        os.makedirs(f'{parent_folder}/{dataset}')
    # if not os.path.exists(profile_file):
    headers = ['annotator_id', 'profile']
    df = pd.DataFrame(columns=headers)
    df.to_csv(profile_file, index=False)
    # if not os.path.exists(log_file):
    headers = ['annotator_id', 'text_ids']
    df = pd.DataFrame(columns=headers)
    df.to_csv(log_file, index=False)

    # load data
    if os.path.isfile(f'user_context/load_{dataset}.py'):
        # load_dataset = getattr(f'load_{dataset}', f'load_{dataset}')
        df = eval(f'load_{dataset}()').reset_index()
    else:
        df = pd.concat([pd.read_csv(filename) for filename in filenames])
    if 'annotator_id' in df.columns: # classification
        user_id_col = 'annotator_id'
        text_id_col = 'text_id'
        prompt = classification_prompt
    elif 'UserId' in df.columns: # stackexchange
        user_id_col = 'UserId'
        text_id_col = 'QuestionId'
        prompt = generative_prompt
    elif 'user_id' in df.columns: # goodreads
        user_id_col = 'user_id'
        text_id_col = 'book_id'
        prompt = generative_prompt
    else:
        logger.info(f'Skipping {dataset}')
        continue

    logger.info(f"{len(df[user_id_col].unique())} annotators")
    try:
        df[user_id_col] = df[user_id_col].astype(int)
    except:
        logger.info(f'Could not convert {user_id_col} in {dataset} to int, continuing...')

    # load dataset description
    with open(f"user_context/data_descriptions/{dataset}.txt", "r") as file:
        data_description = file.read()
    # only for sample of first N annotators
    # first_n_groups = df[user_id_col].unique()[:num_groups]
    # df = df[df[user_id_col].isin(first_n_groups)]
    # only M samples for each annotator
    df = df.groupby(user_id_col).apply(lambda x: x.sample(min([num_samples,len(x)]))).reset_index(drop=True)#df[df['annotator_id'].isin(first_n_groups)]
    # for each annotator, generate a profile
    # Do in batches of annotators
    batches = split_into_batches(df.groupby(user_id_col), batch_size=batch_size)
    for df_batch in batches:
        # df_sample = df_batch.sample(n=min([num_samples,len(df_batch)]))
        annotator_ids = [str(id) for id in df_batch[user_id_col].unique()]
        logger.info(annotator_ids)
        samples = df_batch.groupby(user_id_col)[text_id_col].apply(lambda x: x.tolist()).reset_index()

        with open(log_file, 'a') as f:
            samples.to_csv(f, header=False, index=False)
        # logger.info('Querying OpenAI')
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                prompt.format(dataset=dataset, batch_size=batch_size, data_description=data_description, user_id_col=user_id_col, text_id_col=text_id_col)
                },
                {
                    "role": "user",
                    "content": f"{df_batch}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "user_profiles",
                    "schema": {
                        "type": "object",
                        "properties": {
                            id:
                            {"type": "string",
                            "description": f"Profile for user {id}"
                            } for id in annotator_ids
                        },
                        "required": annotator_ids,
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        # logger.info('Received OpenAI response')
        # logger.info(completion.choices[0].message.content)
        response_rows = pd.DataFrame.from_dict(json.loads(completion.choices[0].message.content), orient='index').reset_index()

        with open(profile_file, 'a') as f:
            response_rows.to_csv(f, header=False, index=False)