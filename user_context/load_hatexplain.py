import os
import pandas as pd
import json

def load_hatexplain() -> pd.DataFrame:
    with open('data/hatexplain/dataset.json') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index(drop=True)

    df_exploded = df.explode('annotators')

    df_melted = df_exploded.melt(id_vars=['post_id'], value_vars=['annotators'], 
                        var_name='foo', value_name='annotation').drop('foo',axis=1)

    df_norm = pd.json_normalize(df_melted['annotation'])

    result_df = df_melted.join(df_norm).drop('annotation',axis=1)
    result_df = pd.merge(result_df,df[['post_id','post_tokens','rationales']],how='inner',on='post_id').rename({'post_id':'text_id','post_tokens':'text'},axis=1).set_index(['annotator_id','text_id']).sort_index()
    
    return result_df