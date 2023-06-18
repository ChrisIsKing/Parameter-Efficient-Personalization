import math
from os.path import join as os_join
from typing import Dict
from logging import Logger

from transformers import Trainer, AdapterTrainer, TrainingArguments, TrainerCallback
from torch.utils.tensorboard import SummaryWriter

from stefutil import *


_all__ = ['TqdmPostfixCallback', 'MyAdapterTrainer']


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
