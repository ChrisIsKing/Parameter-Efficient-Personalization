import os
import pandas as pd
import json

df_annotations = pd.read_csv('data/wikidetox/aggression_annotations.tsv', sep='\t')
df_comments = pd.read_csv('data/wikidetox/aggression_annotated_comments.tsv', sep='\t')
df = pd.merge(df_annotations, df_comments, how='inner', on='rev_id').rename({'worker_id':'annotator_id','rev_id':'text_id','comment':'text'}, axis=1).set_index(['annotator_id','text_id']).sort_index()
df.to_csv('user_context/data/wikidetox/labels.csv')