import os
import re
import random
from os.path import join as os_join
from typing import List, Union

import numpy as np
import torch

from stefutil import *
from peft_u.util.project_paths import BASE_PATH, PROJ_DIR, PKG_NM, DSET_DIR, MODEL_DIR


__all__ = [
    'sconfig', 'u', 'save_fig',
    'set_seed',
    'on_great_lakes', 'get_base_path',
    'sort_user_ids'
]


sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR)
save_fig = u.save_fig


def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


_USER_IDS = Union[List[str], List[int]]

# For `SubjectiveDiscourse`, user ids are like `worker_50`
sub_dis_pattern = re.compile(r'^worker_(?P<id>\d+)$')


def sort_user_ids(uids: _USER_IDS) -> _USER_IDS:
    if all(isinstance(uid, int) for uid in uids):
        return sorted(uids)
    else:
        assert all(isinstance(uid, str) for uid in uids)
        if all(uid.isdigit() for uid in uids):
            sort_fn = int
        else:
            def sort_fn(x):
                match = sub_dis_pattern.match(x)
                assert match is not None
                return int(match.group('id'))
        return sorted(uids, key=sort_fn)
