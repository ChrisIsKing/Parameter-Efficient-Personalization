import os
from os.path import join as os_join

import torch

from stefutil import *
from peft_u.util.util import *


__all__ = ['hf_model_name_drop_org', 'hf_custom_model_cache_dir', 'get_hf_cache_dir']


def hf_model_name_drop_org(model_name: str) -> str:
    if '/' in model_name:
        org, model_name = model_name.split('/')
    return model_name


def map_output_dir_nm(
        model_name: str = None, name: str = None, peft_approach: str = None, dataset_name: str = None
):
    model_name = hf_model_name_drop_org(model_name)
    d = dict(md_nm=model_name, ds=dataset_name, peft=peft_approach)
    date = now(fmt='short-date')
    ret = f'{date}_{pl.pa(d)}'
    if name:
        ret = f'{ret}_{name}'
    return ret


def hf_custom_model_cache_dir():
    path = os_join(get_base_path(), '.cache', 'huggingface', 'transformers')
    os.makedirs(path, exist_ok=True)
    return path


def get_hf_cache_dir():
    ret = None
    if on_great_lakes():  # download to scratch folder if on GL to save space
        ret = hf_custom_model_cache_dir()
    return ret
