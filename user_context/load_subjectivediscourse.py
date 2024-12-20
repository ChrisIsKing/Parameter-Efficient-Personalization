import os
import pandas as pd
import json

def load_subjectivediscourse() -> pd.DataFrame:
    df = pd.read_csv('data/subjectivediscourse/expanded_with_features_annotated_questions_responses_gold.csv')
    df = df.rename({'gold_worker':'annotator_id','qa_index':'text_id'}, axis=1).set_index(['annotator_id','text_id']).sort_index()
    return df