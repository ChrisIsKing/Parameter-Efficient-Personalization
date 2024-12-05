import os
import pandas as pd
import json

def load_tweeteval() -> pd.DataFrame:
    df = pd.read_csv('data/tweeteval/annotations_g3.csv')#.set_index(['Annotator','ID'])
    df_melted = pd.melt(df, id_vars=['id'], var_name='annotator_id', value_name='label', value_vars=[f'label_M_{i}' for i in range(1,9)]+[f'label_F_{i}' for i in range(9,12)]+[f'label_M_{i}' for i in range(12,14)]+[f'label_F_{i}' for i in range(14,21)])
    result_df = pd.merge(df_melted,df[['id','text']],how='left',left_on='id',right_on='id').rename({'id':'text_id'}, axis=1).set_index(['annotator_id','text_id']).sort_index()
    return result_df