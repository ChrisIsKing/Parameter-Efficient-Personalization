import os
import pandas as pd
import json

df = pd.read_csv('data/subjectivediscourse/expanded_with_features_annotated_questions_responses_gold.csv')
df = df.rename({'gold_worker':'annotator_id','qa_index':'text_id'}, axis=1).set_index(['annotator_id','text_id']).sort_index()
df.to_csv('user_context/data/subjectivediscourse/labels.csv')