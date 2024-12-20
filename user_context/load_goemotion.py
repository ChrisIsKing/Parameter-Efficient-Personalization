import os
import pandas as pd
import json

def load_goemotion() -> pd.DataFrame:
    df = pd.DataFrame()
    for i in range (1,4):
        df = pd.concat([df,pd.read_csv(f'data/goemotion/goemotions_{i}.csv')])
    df = df.rename({'rater_id':'annotator_id','id':'text_id'},axis=1).set_index(['annotator_id','text_id']).sort_index()
    return df