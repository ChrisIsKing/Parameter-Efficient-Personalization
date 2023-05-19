import os
from os.path import join as os_join

from stefutil import *
from peft_u.util.project_paths import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR


__all__ = ['sconfig', 'u', 'save_fig']


sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR)
u.tokenizer_path = os_join(u.base_path, u.proj_dir, 'tokenizers')
os.makedirs(u.tokenizer_path, exist_ok=True)
save_fig = u.save_fig
