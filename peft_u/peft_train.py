import os
import json
from os.path import join as os_join
from typing import Tuple
from logging import Logger
from argparse import ArgumentParser
from functools import partial


import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, PrefixTuningConfig, get_peft_model, PeftConfig
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_linear_schedule_with_warmup,
    PreTrainedTokenizer
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from stefutil import *
from data.utils import set_seed, process_data, instructions, InputEgDataset
from peft_u.util import *


logger = get_logger('PEFT Train')


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")

    train_parser.add_argument("--model", type=str, required=False, default="bigscience/mt0-xxl")
    train_parser.add_argument("--data", type=str, required=True)
    train_parser.add_argument("--task", type=str, required=True, choices=list(instructions.keys()))
    train_parser.add_argument("--method", type=str, required=False, default="lora", choices=["lora", "prefix"])
    train_parser.add_argument("--batch_size", type=int, required=False, default=8)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=3)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--seed", type=int, required=False, default=42)
    train_parser.add_argument("--device", type=str, required=False, default="cuda")
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    train_parser.add_argument('--personalize', type=bool, default=True)

    test_parser.add_argument("--model", type=str, required=False, default="bigscience/mt0-xxl")
    test_parser.add_argument("--data", type=str, required=True)
    test_parser.add_argument("--batch_size", type=int, required=False, default=8)
    test_parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def map_output_dir_nm(
        model_name: str = None, name: str = None, peft_approach: str = None, dataset_name: str = None
):
    if '/' in model_name:
        org, model_name = model_name.split('/')
    d = dict(md_nm=model_name, peft=peft_approach[0], ds=dataset_name)
    date = now(fmt='short-date')
    ret = f'{date}_{pl.pa(d)}'
    if name:
        ret = f'{ret}_{name}'
    return ret


def load_model_n_tokenizer(
        model_name_or_path: str, peft_config: PeftConfig = None, device: str = 'cuda', verbose: bool = False,
        logger_fl: Logger = None
) -> Tuple[torch.nn.Module, PreTrainedTokenizer]:
    cache_dir = None
    if on_great_lakes():  # Save to scratch folder if on GL
        cache_dir = hf_custom_model_cache_dir()
        logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')
        if logger_fl:
            logger_fl.info(f'Loading model {model_name_or_path} with cache dir {cache_dir}... ')

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if verbose:
        logger.info('Loading Peft model...')
    model = get_peft_model(model, peft_config)
    model_meta = dict(param=get_trainable_param_meta(model), size=get_model_size(model))
    if verbose:
        logger.info(f'Model info: {pl.i(model_meta)}')
        if logger_fl:
            logger_fl.info(f'Model info: {model_meta}')

    if verbose:
        logger.info(f'Moving model to {pl.i(device)}...')
    model.to(device)
    return model, tokenizer


def smart_batching_collate(batch, tokenizer):
    """Collate function for PyTorch DataLoader."""
    inputs = [example.process_template() for example in batch]
    targets = [example.process_target() for example in batch]
    tokenizer.model_max_length = 512  # TODO: why not set?
    # TODO: why truncation false?
    batch_encoding = tokenizer(inputs, truncation=False, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, truncation=True, padding="max_length", return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    batch_encoding["labels"] = labels
    return batch_encoding


def train_single(
        model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: InputEgDataset,
        device: str = 'cuda',
        batch_size: int = 8, num_epochs: int = 3, learning_rate: float = 2e-5, weight_decay: float = 0.01,
        output_path: str = None, verbose: bool = False
):
    def _save(dir_nm: str):
        model.save_pretrained(os_join(output_path, dir_nm))
        tokenizer.save_pretrained(os_join(output_path, dir_nm))
        if verbose:
            logger.info(f'Model and tokenizer saved to {pl.i(output_path)}')

    collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
    tr, vl, ts = dataset.train, dataset.val, dataset.test
    train_dataloader = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(vl, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(ts, batch_size=batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_step_per_epoch = len(train_dataloader)
    n_step = (n_step_per_epoch * num_epochs)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(n_step * 0.1),
        num_training_steps=n_step,
    )

    logger_fl = get_logger('PEFT Train', kind='file-write', file_path=os_join(output_path, 'train.log'))
    tb_writer = SummaryWriter(os_join(output_path, f'tensorboard'))

    pret = MlPrettier(ref=dict(step=n_step_per_epoch, epoch=num_epochs), metric_keys=['cls_acc'])
    ls = LogStep(prettier=pret, logger=logger, file_logger=logger_fl, tb_writer=tb_writer)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs+1):
        model.train()
        total_tr_loss = 0

        tr_desc = f'Train Epoch {pl.i(epoch)}/{pl.i(num_epochs)}'
        it = tqdm(enumerate(train_dataloader, start=1), desc=tr_desc, total=n_step_per_epoch)
        for step, batch in it:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_item = loss.detach().item()
            total_tr_loss += loss_item
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            d_log = dict(
                epoch=epoch,
                step=(epoch-1) * n_step_per_epoch + step,
                lr=optimizer.param_groups[0]['lr'],
                loss=loss_item,
            )
            ls.pbar = it
            ls(d_log, training=True, to_console=False)

        model.eval()
        cum_eval_loss = 0
        eval_preds = []
        vl_desc = f'Eval Epoch {pl.i(epoch)}/{pl.i(num_epochs)}'
        it = tqdm(enumerate(val_dataloader), desc=vl_desc, total=len(val_dataloader))
        for step, batch in it:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            cum_eval_loss += outputs.loss.detach().item()

            decoded = tokenizer.batch_decode(
                torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True
            )
            eval_preds.extend(decoded)

        eval_epoch_loss = cum_eval_loss / len(val_dataloader)
        eval_ppl = np.exp(eval_epoch_loss)

        n_correct = 0  # Eval classification accuracy
        n = 0
        for pred, true in zip(eval_preds, vl):
            if pred.strip() == true.process_target().strip():
                n_correct += 1
            n += 1

        d_log = dict(epoch=epoch, eval_loss=eval_epoch_loss, eval_ppl=eval_ppl, eval_cls_acc=n_correct/n)
        ls(d_log, training=False, to_console=False)

        train_epoch_loss = total_tr_loss / len(train_dataloader)
        d_log.update(dict(train_epoch_loss=train_epoch_loss, train_ppl=np.exp(train_epoch_loss)))
        logger.info(pl.i(ls.prettier(d_log)))

        _save(f'epoch_{epoch:02d}')
        if eval_epoch_loss < best_val_loss:
            best_val_loss = eval_epoch_loss
            _save('trained')
            logger.info(f'Best model saved w/ eval loss {pl.i(best_val_loss)}')
            logger_fl.info(f'Best model saved w/ eval loss {best_val_loss}')


if __name__ == '__main__':
    def run_train():
        args = parse_args()
        if args.mode == 'train':
            batch_size = args.batch_size
            num_epochs = args.num_epochs
            learning_rate = args.learning_rate
            weight_decay = args.weight_decay
            model_name_or_path = args.model
            data_path = args.data
            method = args.method
            output_dir = args.output_dir
            seed = args.seed
            task = args.task
            device = args.device
            personalize = args.personalize

            map_args = dict(model_name=model_name_or_path, name=output_dir, peft_approach=method, dataset_name=task)
            output_path = os_join(get_base_path(), u.proj_dir, u.model_dir, map_output_dir_nm(**map_args))
            os.makedirs(output_path, exist_ok=True)

            set_seed(seed)
            d_log = dict(
                batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                model_name_or_path=model_name_or_path, data_path=data_path, method=method,
                seed=seed, task=task, device=device, personalize=personalize,
                output_dir=output_dir, output_path=output_path
            )
            logger_fl = get_logger('PEFT Train', kind='file-write', file_path=os_join(output_path, 'train.log'))
            logger.info(f'Training PEFT w/ {pl.i(d_log)}...')
            logger_fl.info(f'Training PEFT w/ {d_log}...')

            if method == 'lora':
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
                )
            else:
                assert method == 'prefix'
                peft_config = PrefixTuningConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20
                )

            data_path = os_join(u.proj_path, 'data', data_path)
            logger.info(f'Loading data from {pl.i(data_path)}...')
            with open(data_path, 'r') as f:
                data = json.load(f)

            dset = process_data(data, task)
            load_args = dict(peft_config=peft_config, device=device, logger_fl=logger_fl)
            if personalize:
                for uid in sorted(dset.keys(), key=int):  # TODO: for `tweeteval`, uid is integer
                    logger.info(f'Launching personalized training for User {pl.i(uid)}...')

                    # reload model for each user
                    model, tokenizer = load_model_n_tokenizer(model_name_or_path, **load_args)
                    train_single(
                        model=model, tokenizer=tokenizer, dataset=dset[uid], device=device,
                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                        weight_decay=weight_decay, output_path=os_join(output_path, f'User-{uid}')
                    )
                    # raise NotImplementedError
            else:
                model, tokenizer = load_model_n_tokenizer(model_name_or_path, **load_args, verbose=True)
                train_single(
                    model=model, tokenizer=tokenizer, dataset=dset, device=device,
                    batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                    weight_decay=weight_decay, output_path=output_path
                )
    run_train()
