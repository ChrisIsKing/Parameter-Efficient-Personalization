import os
import json
from os.path import join as os_join
from typing import Tuple, List
from logging import Logger
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, PrefixTuningConfig, get_peft_model, PeftConfig, PeftModel
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_linear_schedule_with_warmup,
    PreTrainedTokenizer
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from stefutil import *
from peft_u.util import *
from peft_u.data.utils import *


logger = get_logger('PEFT Train')


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")

    default_md_nm = 'google/flan-t5-base'
    dataset_names = list(sconfig('datasets'))

    train_parser.add_argument("--model", type=str, required=False, default=default_md_nm)
    train_parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    train_parser.add_argument("--leakage", type=bool, required=False, default=False)
    train_parser.add_argument("--method", type=str, required=False, default="lora", choices=["lora", "prefix"])
    train_parser.add_argument("--batch_size", type=int, required=False, default=8)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=3)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--seed", type=int, required=False, default=42)
    train_parser.add_argument("--device", type=str, required=False, default="cuda")
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    train_parser.add_argument('--personalize', type=bool, default=True)

    test_parser.add_argument("--model", type=str, required=False, default=default_md_nm)
    test_parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    test_parser.add_argument("--leakage", type=str, required=False, default=False)
    test_parser.add_argument("--batch_size", type=int, required=False, default=8)
    test_parser.add_argument("--output_dir", type=str, required=True)
    test_parser.add_argument('--personalize', type=bool, default=True)

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
        model_name_or_path: str, peft_method: str = None, device: str = 'cuda', verbose: bool = False,
        logger_fl: Logger = None
) -> Tuple[PeftModel, PreTrainedTokenizer]:
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

    if peft_method == 'lora':
        peft_config: PeftConfig = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
    else:
        assert peft_method == 'prefix'
        peft_config: PeftConfig = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20
        )
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
    """
    Collate function for PyTorch DataLoader
    """
    inputs = [example.process_template() for example in batch]
    targets = [example.process_target() for example in batch]
    tokenizer.model_max_length = 512
    batch_encoding = tokenizer(inputs, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(targets, truncation=True, padding='max_length', return_tensors='pt')
    labels = labels['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    batch_encoding['labels'] = labels
    return batch_encoding


class ListDataset(Dataset):
    def __init__(self, lst: List):
        self.lst = lst

    def __getitem__(self, idx):
        return self.lst[idx]

    def __len__(self):
        return len(self.lst)


def train_single(
        model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: InputEgDataset,
        device: str = 'cuda', seed: int = 42,
        batch_size: int = 8, num_epochs: int = 3, learning_rate: float = 2e-5, weight_decay: float = 0.01,
        output_path: str = None, verbose: bool = False
):
    def _save(dir_nm: str):
        model.save_pretrained(os_join(output_path, dir_nm))
        tokenizer.save_pretrained(os_join(output_path, dir_nm))
        if verbose:
            logger.info(f'Model and tokenizer saved to {pl.i(output_path)}')

    set_seed(seed)
    collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
    tr, vl, ts = dataset.train, dataset.val, dataset.test

    train_dataloader = DataLoader(ListDataset(tr), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(ListDataset(vl), batch_size=batch_size, collate_fn=collate_fn)

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
    d_log = dict(train=len(tr), val=len(vl), test=len(ts))
    logger.info(f'Dataset sizes: {pl.i(d_log)}')
    logger_fl.info(f'Dataset sizes: {d_log}')

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

        n_vl_step = len(val_dataloader)
        it = tqdm(enumerate(val_dataloader), desc=vl_desc, total=n_vl_step)

        eval_epoch_loss = None
        for step, batch in it:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            cum_eval_loss += outputs.loss.detach().item()

            decoded = tokenizer.batch_decode(
                torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True
            )
            eval_preds.extend(decoded)

            if step + 1 == n_vl_step:  # last iteration
                eval_epoch_loss = cum_eval_loss / len(val_dataloader)
                eval_ppl = np.exp(eval_epoch_loss)

                n_correct = 0  # Eval classification accuracy
                n = 0
                for pred, true in zip(eval_preds, vl):
                    if pred.strip() == true.process_target().strip():
                        n_correct += 1
                    n += 1

                train_epoch_loss = total_tr_loss / len(train_dataloader)
                d_log = dict(
                    epoch=epoch, eval_loss=eval_epoch_loss, eval_ppl=eval_ppl, eval_cls_acc=n_correct/n,
                    train_epoch_loss=train_epoch_loss, train_ppl=np.exp(train_epoch_loss)
                )
                ls.pbar = it
                ls(d_log, training=False, to_console=False)
        assert eval_epoch_loss is not None  # sanity check

        _save(f'epoch_{epoch:02d}')
        if eval_epoch_loss < best_val_loss:
            best_val_loss = eval_epoch_loss
            _save('trained')

            best_val_loss_ = round(best_val_loss, 4)
            logger.info(f'Best model saved w/ eval loss {pl.i(best_val_loss_)}')
            logger_fl.info(f'Best model saved w/ eval loss {best_val_loss_}')


def load_trained(model_name_or_path: str = None) -> Tuple[PeftModel, PreTrainedTokenizer]:
    config = PeftConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    def run():
        args = parse_args()
        cmd = args.mode
        if cmd == 'train':
            model_name_or_path, method = args.model, args.method
            dataset_name, leakage, personalize = args.dataset_name, args.leakage, args.personalize
            batch_size, num_epochs, learning_rate = args.batch_size, args.num_epochs, args.learning_rate
            weight_decay = args.weight_decay
            seed, device = args.seed, args.device
            output_dir = args.output_dir

            map_args = dict(model_name=model_name_or_path, name=output_dir, peft_approach=method)
            out_dir_nm = map_output_dir_nm(**map_args, dataset_name=dataset_name)
            output_path = os_join(get_base_path(), u.proj_dir, u.model_dir, out_dir_nm)
            os.makedirs(output_path, exist_ok=True)

            d_log = dict(
                batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                model_name_or_path=model_name_or_path, dataset_name=dataset_name, leakage=leakage, method=method,
                seed=seed, device=device, personalize=personalize,
                output_dir=output_dir, output_path=output_path
            )
            logger_fl = get_logger('PEFT Train', kind='file-write', file_path=os_join(output_path, 'train.log'))
            logger.info(f'Training PEFT w/ {pl.i(d_log)}...')
            logger_fl.info(f'Training PEFT w/ {d_log}...')

            dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage)
            load_args = dict(device=device, logger_fl=logger_fl, method=method)
            if personalize:
                uids = list(dset.keys())
                sort_fn = int if all(uid.isdigit() for uid in uids) else None
                for uid in sorted(uids, key=sort_fn):
                    logger.info(f'Launching personalized training for User {pl.i(uid)}...')

                    # reload model for each user
                    model, tokenizer = load_model_n_tokenizer(model_name_or_path, **load_args)
                    train_single(
                        model=model, tokenizer=tokenizer, dataset=dset[uid], device=device, seed=seed,
                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                        weight_decay=weight_decay, output_path=os_join(output_path, f'User-{uid}')
                    )
            else:
                model, tokenizer = load_model_n_tokenizer(model_name_or_path, **load_args, verbose=True)
                train_single(
                    model=model, tokenizer=tokenizer, dataset=dset, device=device,
                    batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                    weight_decay=weight_decay, output_path=output_path
                )
        else:
            assert cmd == 'test'
            model_name_or_path = args.model

            dataset_name, leakage, personalize = args.dataset_name, args.leakage, args.personalize
            bsz = args.batch_size
            output_dir = args.output_dir

            if personalize:
                dset = load_dataset_with_prompts(dataset_name, leakage=leakage)

                model_name_or_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_name_or_path)
                model, tokenizer = load_trained(model_name_or_path=model_name_or_path)

                collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
                ts_dl = DataLoader(ListDataset(dset.test), batch_size=bsz, collate_fn=collate_fn)

                it = tqdm(ts_dl, desc=f'Testing on {pl.i(dataset_name)}')
                for ba in it:
                    ba = {k: v for k, v in ba.items()}
                    if torch.cuda.is_available():
                        ba = {k: v.cuda() for k, v in ba.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs)  # Greedy decoding
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                    mic(decoded)
                    raise NotImplementedError
                    eval_preds.extend(decoded)
            else:
                raise NotImplementedError
    run()
