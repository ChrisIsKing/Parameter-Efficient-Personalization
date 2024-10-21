import os
import pandas as pd
import json

# os.chdir('/Users/bradenloughnane/Parameter-Efficient-Personalization')

# Load JSON file
with open('data/hatexplain/dataset.json') as f:
    data = json.load(f)

# Normalize the JSON data
# If your JSON is deeply nested, you can use json_normalize
# print(data['word_ratings']['votes'])
df = pd.DataFrame.from_dict(data, orient='index')#['word_ratings']['votes']])
# df.index.name = 'post_id'
df = df.reset_index(drop=True)

df_exploded = df.explode('annotators')

print(df_exploded.columns)
# print(df_exploded.drop('post_id',axis=1))

df_melted = df_exploded.melt(id_vars=['post_id'], value_vars=['annotators'], 
                    var_name='foo', value_name='annotation').drop('foo',axis=1)

df_norm = pd.json_normalize(df_melted['annotation'])

result_df = df_melted.join(df_norm).drop('annotation',axis=1)
result_df = pd.merge(result_df,df[['post_id','post_tokens','rationales']],how='inner',on='post_id').rename({'post_id':'text_id','post_tokens':'text'},axis=1).set_index(['annotator_id','text_id']).sort_index()

print(result_df)

# # Explode the list of id to create one row per id
# df_exploded = df_melted.explode('id').dropna(subset=['id'])

# # Create a boolean column based on the 'vote_type'
# df_exploded['vote_yes'] = df_exploded['vote_type'] == 'yes_votes'

# # Drop the 'vote_type' column and set 'id' as the index
# result_df = df_exploded.drop(columns=['vote_type']).set_index(['id','category'])

# # result_df = result_df.sort_values('id')
# result_df = result_df.sort_index()

result_df.to_csv('user_context/data/hatexplain/labels.csv')

# Display the DataFrame
# print(result_df.columns)
# print(result_df.head())
# print(result_df.iloc[37])
# print(result_df.groupby('id').count())