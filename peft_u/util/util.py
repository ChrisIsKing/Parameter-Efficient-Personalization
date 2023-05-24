import os
import random
from os.path import join as os_join

import numpy as np
import torch

from stefutil import *
from peft_u.util.project_paths import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR


__all__ = [
    'sconfig', 'u', 'save_fig',
    'set_seed', 'flip_dict_of_lists',
    'on_great_lakes', 'get_base_path',
    'hf_custom_model_cache_dir', 'get_trainable_param_meta', 'get_model_size'
]


sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR)
u.tokenizer_path = os_join(u.base_path, u.proj_dir, 'tokenizers')
os.makedirs(u.tokenizer_path, exist_ok=True)
save_fig = u.save_fig


def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(0)


def flip_dict_of_lists(d):
    """
    Flip a dictionary of lists.
    """
    return {v: k for k, values in d.items() for v in values}


def get_current_directory():
    """
    Get the current directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def on_great_lakes():
    return 'arc-ts' in get_hostname()


def get_base_path():
    # For remote machines, save heavy-duty data somewhere else to save `/home` disk space
    hnm = get_hostname()
    if 'clarity' in hnm:  # Clarity lab
        return '/data'
    elif on_great_lakes():  # Great Lakes; `profmars0` picked arbitrarily among [`profmars0`, `profmars1`]
        # Per https://arc.umich.edu/greatlakes/user-guide/
        return os_join('/scratch', 'profmars_root', 'profmars0', 'stefanhg')
    else:
        return u.base_path


def hf_custom_model_cache_dir():
    path = os_join(get_base_path(), '.cache', 'huggingface', 'transformers')
    os.makedirs(path, exist_ok=True)
    return path


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
