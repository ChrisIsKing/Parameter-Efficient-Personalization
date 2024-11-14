import os
import pandas as pd
import json

def load_epic() -> pd.DataFrame:
    df = pd.read_csv('data/epic/EPICorpus.csv').rename({'user':'annotator_id','id_original':'text_id'},axis=1).set_index(['annotator_id','text_id']).sort_index()
    return df