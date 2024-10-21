import os
import pandas as pd
import json

# os.chdir('/Users/bradenloughnane/Parameter-Efficient-Personalization')

# Normalize the JSON data
# If your JSON is deeply nested, you can use json_normalize
# print(data['word_ratings']['votes'])
# df = pd.concat([pd.DataFrame.from_dict(d, orient='index') for d in data['word_ratings']['votes']])
df = pd.read_csv('data/epic/EPICorpus.csv').rename({'user':'annotator_id','id_original':'text_id'},axis=1).set_index(['annotator_id','text_id']).sort_index()
# df.index.name = 'category'
# df = df.reset_index()


# df = df.

# print(df)
# df_melted = df.melt(id_vars=['category'], value_vars=['yes_votes', 'no_votes'], 
#                     var_name='vote_type', value_name='id')

# # Explode the list of id to create one row per id
# df_exploded = df_melted.explode('id').dropna(subset=['id'])

# # Create a boolean column based on the 'vote_type'
# df_exploded['vote_yes'] = df_exploded['vote_type'] == 'yes_votes'

# # Drop the 'vote_type' column and set 'id' as the index
# result_df = df_exploded.drop(columns=['vote_type']).set_index('id')

# result_df = result_df.sort_values('category')
# result_df = result_df.sort_index()

df.to_csv('user_context/data/epic/labels.csv')

# Display the DataFrame
# print(result_df.columns)
# print(result_df.head())
# print(result_df.iloc[37])
# print(result_df.groupby('id').count())