import os
import pandas as pd
import json

df = pd.read_csv('data/gabhate/GabHateCorpus_annotations.tsv', sep='\t').rename({'Annotator':'annotator_id','ID':'text_id','Text':'text'},axis=1).set_index(['annotator_id','text_id']).sort_index()
df.to_csv('user_context/data/gabhate/labels.csv')