from openai import OpenAI
client = OpenAI()

import os
import csv
import pandas as pd
import itertools
import json

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

num_groups = 100
num_samples = 20
batch_size = 1

datasets = [f for f in os.listdir('user_context/data') if os.path.isdir(os.path.join('user_context/data', f))]
parent_folder = f'user_context/profiles_n{batch_size}/'

for dataset in datasets:
    print(f'\n\n{dataset}:')
    # read content from a file
    file_path = f'user_context/data/{dataset}/labels.csv'
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
    df = pd.read_csv(file_path)
    print(f"{len(df['annotator_id'].unique())} annotators")

    # load dataset description
    with open(f"user_context/data_descriptions/{dataset}.txt", "r") as file:
        data_description = file.read()
    # only for sample of first N annotators
    first_n_groups = df['annotator_id'].unique()[:num_groups]
    df = df[df['annotator_id'].isin(first_n_groups)]
    # only M samples for each annotator
    df = df.groupby('annotator_id').apply(lambda x: x.sample(min([num_samples,len(x)]))).reset_index(drop=True)#df[df['annotator_id'].isin(first_n_groups)]
    # for each annotator, generate a profile
    # Do in batches of annotators
    batches = split_into_batches(df.groupby('annotator_id'), batch_size=batch_size)
    for df_batch in batches:
        # df_sample = df_batch.sample(n=min([num_samples,len(df_batch)]))
        annotator_ids = [str(id) for id in df_batch['annotator_id'].unique()]
        print(annotator_ids)
        samples = df_batch.groupby('annotator_id')['text_id'].apply(lambda x: x.tolist()).reset_index()

        with open(log_file, 'a') as f:
            samples.to_csv(f, header=False, index=False)
        # print('abc123\n',df_batch)
        # print('Querying OpenAI')
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                f"""
                Attached are sub-samples of the {dataset} dataset for {batch_size} annotators (identified by the column 'annotator_id'). The dataset consists of labeled classification data. A description of the dataset is provided below:

                Dataset description:
                {data_description}
                
                Perform the following steps and return a annotator profile ({{profile}}).
                
                (1) Determine the schema of the dataset:
                    (1.a) The columns 'annotator_id' and 'text_id' are the IDs for the annotator and the data to be labeled, respectively
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
                },
                {
                    "role": "user",
                    "content": f"{df_batch}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "annotator_profiles",
                    "schema": {
                        "type": "object",
                        "properties": {
                            id:
                            {"type": "string",
                            "description": f"Profile for annotator {id}"
                            } for id in annotator_ids
                        },
                        "required": annotator_ids,
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        # print('Received OpenAI response')
        # print(completion.choices[0].message.content)
        response_rows = pd.DataFrame.from_dict(json.loads(completion.choices[0].message.content), orient='index').reset_index()

        with open(profile_file, 'a') as f:
            response_rows.to_csv(f, header=False, index=False)