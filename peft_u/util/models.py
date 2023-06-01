import os
from os.path import join as os_join

import torch

from stefutil import *
from peft_u.util.util import *


__all__ = [
    'get_trainable_param_meta', 'get_model_size',
    'hf_model_name_drop_org', 'hf_custom_model_cache_dir', 'get_hf_cache_dir'
]


def get_trainable_param_meta(model: torch.nn.Module, fmt='str'):
    """
    Edited from `PeftModel.get_trainable_parameters`
    """
    ca.check_mismatch('#Param Format', fmt, ['int', 'str'])

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    ratio = round(trainable_params / all_param * 100, 2)
    if fmt == 'str':
        trainable_params = fmt_num(trainable_params)
        all_param = fmt_num(all_param)
    return {
        '#trainable': trainable_params,
        '#all': all_param,
        'ratio': f'{ratio}%'
    }


def get_model_size(model: torch.nn.Module, fmt='str', all_only: bool = True):
    ca.check_mismatch('Size Format', fmt, ['int', 'str'])

    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    ret = dict(param_size=param_size, buffer_size=buffer_size, size_all=param_size + buffer_size)
    if fmt == 'str':
        ret = {k: fmt_sizeof(v) for k, v in ret.items()}
    return ret['size_all'] if all_only else ret


def hf_model_name_drop_org(model_name: str) -> str:
    if '/' in model_name:
        org, model_name = model_name.split('/')
    return model_name


def map_output_dir_nm(
        model_name: str = None, name: str = None, peft_approach: str = None, dataset_name: str = None
):
    model_name = hf_model_name_drop_org(model_name)
    d = dict(md_nm=model_name, peft=peft_approach, ds=dataset_name)
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
