import os
from os.path import join as os_join
from typing import Dict
from logging import Logger

import torch

from stefutil import *
from peft_u.util.util import *


__all__ = [
    'hf_model_name_drop_org', 'get_train_output_path', 'prepend_local_model_path',
    'hf_custom_model_cache_dir', 'hf_custom_dataset_cache_dir', 'get_hf_model_cache_dir', 'get_hf_dataset_cache_dir',
    'get_cuda_free_mem',
    'LoadModelLogging'
]


def hf_model_name_drop_org(model_name: str) -> str:
    if '/' in model_name:
        org, model_name = model_name.split('/')
    return model_name


def map_output_dir_nm(
        model_name: str = None, name: str = None, method: str = None, dataset_name: str = None,
        method_key: str = None
) -> str:
    d = dict()
    if method_key is not None:
        d[method_key] = method
    d |= dict(md_nm=hf_model_name_drop_org(model_name), ds=dataset_name)
    date = now(fmt='short-date')
    ret = f'{date}_{pl.pa(d, omit_none_val=True)}'
    if name:
        ret = f'{ret}_{name}'
    return ret


def get_train_output_path(**kwargs) -> str:
    out_dir_nm = map_output_dir_nm(**kwargs)
    output_path = os_join(get_base_path(), u.proj_dir, u.model_dir, out_dir_nm)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def prepend_local_model_path(model_path: str) -> str:
    ret = os_join(get_base_path(), u.proj_dir, u.model_dir, model_path)
    if not os.path.exists(ret):
        ret = model_path  # reset
    return ret


def hf_custom_model_cache_dir():
    path = os_join(get_base_path(), '.cache', 'huggingface', 'transformers')
    os.makedirs(path, exist_ok=True)
    return path


def hf_custom_dataset_cache_dir():
    path = os_join(get_base_path(), '.cache', 'huggingface', 'datasets')
    os.makedirs(path, exist_ok=True)
    return path


def get_hf_model_cache_dir():
    ret = None
    if on_great_lakes():  # download to scratch folder if on GL to save space
        ret = hf_custom_model_cache_dir()
    return ret


def get_hf_dataset_cache_dir():
    ret = None
    if on_great_lakes():
        ret = hf_custom_dataset_cache_dir()
    return ret


def get_cuda_free_mem() -> Dict:
    free, total = torch.cuda.mem_get_info(device='cuda:0')
    ratio = free / total * 100
    ratio_str = f'{ratio:.2f}%'
    return dict(free_sz=fmt_sizeof(free), total_sz=fmt_sizeof(total), free_ratio=ratio_str)


class LoadModelLogging:
    def __init__(self, logger: Logger = None, logger_fl: Logger = None, verbose: bool = False):
        self.logger = logger
        self.logger_fl = logger_fl
        self.verbose = verbose

    def get_n_log_cache_dir(self, model_name_or_path: str = None) -> str:
        cache_dir = get_hf_model_cache_dir()
        if self.verbose and self.logger:
            self.logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')
        if self.logger_fl:
            self.logger_fl.info(f'Loading model {model_name_or_path} with cache dir {cache_dir}... ')
        return cache_dir

    def get_n_log_model_meta(self, model: torch.nn.Module):
        model_meta = get_model_meta(model)
        if self.verbose and self.logger:
            self.logger.info(f'Model info: {pl.i(model_meta)}')
        if self.logger_fl:
            self.logger_fl.info(f'Model info: {model_meta}')
        return model_meta
