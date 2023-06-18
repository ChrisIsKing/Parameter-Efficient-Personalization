import json
import math
from os.path import join as os_join
from typing import Dict, List
from logging import Logger

import numpy as np
from transformers import Trainer, AdapterTrainer, TrainingArguments, TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stefutil import *


_all__ = [
    'TqdmPostfixCallback', 'MyAdapterTrainer',
    'get_user_str_w_ordinal', 'get_user_test_pbar', 'test_user_update_postfix_n_write_df', 'log_n_save_test_results'
]


logger = get_logger('Train Util')


class TqdmPostfixCallback(TrainerCallback):
    def __init__(self, trainer: Trainer = None, logger_fl: Logger = None):
        args = trainer.args
        n_ep = args.num_train_epochs
        bsz = args.per_device_train_batch_size * args.gradient_accumulation_steps
        n_data = len(trainer.train_dataset)
        n_step = max(math.ceil(n_data / bsz), 1) * n_ep
        mp = MlPrettier(ref=dict(step=n_step, epoch=n_ep))

        writer = SummaryWriter(os_join(trainer.args.output_dir, 'tensorboard'))
        self.ls = LogStep(trainer=trainer, prettier=mp, logger=logger, file_logger=logger_fl, tb_writer=writer)

    def on_log(self, args: TrainingArguments, state, control, logs: Dict = None, **kwargs):
        step = state.global_step
        if 'loss' in logs:  # training step
            d_log = dict(epoch=state.epoch, step=step+1)  # 1-indexed
            d_log.update(dict(lr=logs['learning_rate'], loss=logs['loss']))
            self.ls(d_log, training=True, to_console=False)
        elif 'eval_loss' in logs:  # eval for each epoch
            n_ep = logs['epoch']
            assert n_ep.is_integer()
            d_log = dict(epoch=int(n_ep), loss=logs['eval_loss'])
            self.ls(d_log, training=False, to_console=False)
        else:
            logger.info(pl.i(logs))


class MyAdapterTrainer(AdapterTrainer):
    def __init__(self, logger_fl: Logger = None, **kwargs):
        super(MyAdapterTrainer, self).__init__(**kwargs)

        callbacks = self.callback_handler.callbacks
        self.callback_handler.callbacks = [  # Remove internal callback
            c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
        ]

        self.add_callback(MyProgressCallback())

        self.add_callback(TqdmPostfixCallback(trainer=self, logger_fl=logger_fl))

    def create_optimizer(self):
        """
        Use the implementation from original HuggingFace Trainer class
        cos the `AdapterTrainer` implementation forces using HF's AdamW
        """
        super(AdapterTrainer, self).create_optimizer()


def get_user_str_w_ordinal(user_id: str = None, user_idx: int = None, n_user: int = None):
    """
    Intended for terminal output
    """
    user_ordinal = f'{pl.i(user_idx)}/{pl.i(n_user)}'
    return f'{pl.i(user_id)}({user_ordinal})'


def get_user_test_pbar(it=None, user_id: str = None, user_idx: int = None, n_user: int = None, **kwargs):
    desc = f'{pl.i(now(for_path=True, color=True))} Testing on User {get_user_str_w_ordinal(user_id, user_idx, n_user)}'
    return tqdm(it, desc=desc, **kwargs)


def test_user_update_postfix_n_write_df(
        label_options: List[str] = None, trues: np.ndarray = None, preds: np.ndarray = None,
        pbar: tqdm = None, d_postfix: Dict = None, df_out_path: str = None
) -> float:
    idx_lbs = list(range(len(label_options)))
    args = dict(
        labels=[-1, *idx_lbs], target_names=['Label not in dataset', *label_options],
        zero_division=0, output_dict=True
    )
    df, acc = eval_array2report_df(labels=trues, preds=preds, report_args=args, pretty=False)
    acc_str = f'{acc * 100:.1f}'
    d_postfix['cls_acc'] = pl.i(acc_str)
    pbar.set_postfix(d_postfix)

    df.to_csv(df_out_path)
    return acc


def log_n_save_test_results(
        d_accs: Dict[str, float] = None, dataset_name: str = None, logger_fl: Logger = None,
        eval_output_path: str = None
):
    acc_avg = np.mean(list(d_accs.values()))
    acc_avg_str = f'{acc_avg * 100:.1f}'
    logger.info(f'Dataset {pl.i(dataset_name)} macro-avg acc: {pl.i(acc_avg_str)}')
    logger_fl.info(f'Dataset {dataset_name} macro-avg acc: {acc_avg_str}')
    with open(os_join(eval_output_path, 'accuracies.json'), 'w') as f:
        json.dump(d_accs, f, indent=4)
