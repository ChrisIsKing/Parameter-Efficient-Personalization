import os
import pandas as pd
import json

# os.chdir('/Users/bradenloughnane/Parameter-Efficient-Personalization')
def load_cockamamie():
    # Load JSON file
    with open('data/cockamamie/cockamamie.json') as f:
        data = json.load(f)

    # Normalize the JSON data
    # If your JSON is deeply nested, you can use json_normalize
    # print(data['word_ratings']['votes'])
    df = pd.concat([pd.DataFrame.from_dict(d, orient='index') for d in data['word_ratings']['votes']])
    df.index.name = 'text'
    df = df.reset_index()

    df_melted = df.melt(id_vars=['text'], value_vars=['yes_votes', 'no_votes'], 
                        var_name='vote_type', value_name='annotator_id')

    # Explode the list of id to create one row per id
    df_exploded = df_melted.explode('annotator_id').dropna(subset=['annotator_id'])

    # Create a boolean column based on the 'vote_type'
    df_exploded['is_humorous'] = df_exploded['vote_type'] == 'yes_votes'

    # Drop the 'vote_type' column and set 'id' as the index
    df_exploded['text_id'] = df_exploded.groupby('text').ngroup()
    result_df = df_exploded.drop(columns=['vote_type']).set_index(['annotator_id','text_id']).sort_index()

    # result_df = result_df.sort_values('id')

    return result_df
    # Display the DataFrame
    # print(result_df.columns)
    # print(result_df.head())
    # print(result_df.iloc[37])
    # print(result_df.groupby('id').count())