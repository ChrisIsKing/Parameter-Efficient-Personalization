import os
import pandas as pd
import json

def load_studemo() -> pd.DataFrame:
    df_annotation = pd.read_csv(f'data/studemo/annotation_data.csv')#.set_index(['annotator_id','text_id'])
    df_text = pd.read_csv(f'data/studemo/text_data.csv')#.set_index(['text_id'])
    df = pd.merge(df_annotation,df_text,how='inner',left_on='text_id',right_on='text_id').set_index(['annotator_id','text_id']).sort_index()

    return df