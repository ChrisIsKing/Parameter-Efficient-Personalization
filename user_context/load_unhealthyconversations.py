import os
import pandas as pd
import json

def load_unhealthyconversations() -> pd.DataFrame:
    df = pd.read_csv('data/unhealthyconversations/unhealthy_full.csv').rename({'_worker_id':'annotator_id','_unit_id':'text_id','comment':'text'}, axis=1).set_index(['annotator_id','text_id']).sort_index()
    return df