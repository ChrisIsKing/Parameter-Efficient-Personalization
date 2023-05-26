import os
from os.path import join as os_join
from typing import Tuple
from logging import Logger
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd
import torch
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
import peft_u.util.models as model_util
from peft_u.preprocess.load_dataset import *


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
    test_parser.add_argument('--personalize', type=bool, default=True)

    return parser.parse_args()


def load_model_n_tokenizer(
        model_name_or_path: str, peft_method: str = None, device: str = 'cuda', verbose: bool = False,
        logger_fl: Logger = None
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    cache_dir = model_util.get_hf_cache_dir()
    if verbose:
        logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')
        if logger_fl:
            logger_fl.info(f'Loading model {model_name_or_path} with cache dir {cache_dir}... ')

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = 512

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
    model_meta = dict(param=model_util.get_trainable_param_meta(model), size=model_util.get_model_size(model))
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
    # mic(inputs, targets)
    # raise NotImplementedError
    batch_encoding = tokenizer(inputs, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(targets, truncation=True, padding='max_length', return_tensors='pt')
    labels = labels['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    batch_encoding['labels'] = labels
    return batch_encoding


def train_single(
        model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: InputEgDataset,
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


def load_trained(model_name_or_path: str = None, verbose: bool = False) -> Tuple[PeftModel, PreTrainedTokenizer]:
    cache_dir = model_util.get_hf_cache_dir()
    if verbose:
        logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')

    config = PeftConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir)
    model = PeftModel.from_pretrained(model, model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


def test_single(
        model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: ListDataset = None, dataset_name: str = None,
        batch_size: int = 8, user_id: str = None
) -> Tuple[pd.DataFrame, float]:
    label_options = sconfig(f'datasets.{dataset_name}.labels')
    lb2id = {lb: i for i, lb in enumerate(label_options)}  # sanity check each pred and true label is in config
    n_sample = len(dataset)

    collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
    ts_dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    trues, preds = np.empty(n_sample, dtype=int), np.empty(n_sample, dtype=int)
    it = tqdm(ts_dl, desc=f'Testing on User {pl.i(user_id)}... ' if user_id else 'Testing... ')
    d_it = dict(dataset_size=pl.i(len(dataset)))
    it.set_postfix(d_it)

    ret = None
    n_ba = len(ts_dl)
    for i_ba, inputs in enumerate(it):
        inputs = {k: v for k, v in inputs.items()}
        inputs.pop('labels')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)  # Greedy decoding
        lst_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, decoded in enumerate(lst_decoded):
            i_sample = i_ba * batch_size + i
            labels = dataset[i_sample].process_target()
            labels = labels.split(', ')  # See `peft_u.preprocess.load_dataset::InputExample.process_target`
            # model may generate multiple labels; enforce an easier requirement by allowing no whitespace between labels
            decoded = [d.strip() for d in decoded.split(',')]

            correct = False
            for dec in decoded:
                if dec in labels:  # predicts one of the correct label, declare correct; TODO: discuss
                    preds[i_sample] = trues[i_sample] = lb2id[dec]
                    correct = True
                    break
            if not correct:  # if prediction wrong, default to first label arbitrarily
                trues[i_sample] = lb2id[labels[0]]
                preds[i_sample] = lb2id[decoded[0]]

        if i_ba + 1 == n_ba:  # last iteration
            idx_lbs = list(range(len(label_options)))
            args = dict(labels=idx_lbs, target_names=label_options, zero_division=0, output_dict=True)
            ret = df, acc = eval_array2report_df(labels=trues, preds=preds, report_args=args, pretty=False)
            acc_str = f'{acc*100:.1f}'
            d_it['cls_acc'] = pl.i(acc_str)
            it.set_postfix(d_it)
    return ret


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
            out_dir_nm = model_util.map_output_dir_nm(**map_args, dataset_name=dataset_name)
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
            load_args = dict(peft_method=method, device=device, logger_fl=logger_fl)
            if personalize:
                for uid in iter_users(dset):
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

            date = now(fmt='short-date')
            eval_output_path = os_join(u.eval_path, f'{model_name_or_path}_Eval-{date}')
            os.makedirs(eval_output_path, exist_ok=True)

            d_log = dict(
                model_name_or_path=model_name_or_path, dataset_name=dataset_name, leakage=leakage,
                personalize=personalize, batch_size=bsz
            )
            logger.info(f'Testing PEFT w/ {pl.i(d_log)}...')

            if personalize:
                dset = load_dataset_with_prompts(dataset_name, leakage=leakage)
                model_name_or_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_name_or_path)

                accs = []

                for uid in iter_users(dset):
                    ts = ListDataset(dset[uid].test)
                    # logger.info(f'Testing personalized PEFT for User {pl.i(uid)} w/ test split size {pl.i(len(ts))}...')

                    user_str = f'User-{uid}'
                    path = os_join(model_name_or_path, user_str, 'trained')
                    assert os.path.exists(path)  # sanity check
                    model, tokenizer = load_trained(model_name_or_path=path)

                    df, acc = test_single(
                        model=model, tokenizer=tokenizer, dataset=ts, batch_size=bsz,
                        dataset_name=dataset_name, user_id=uid
                    )
                    path = os_join(eval_output_path, f'{user_str}.csv')
                    df.to_csv(path)

                    accs.append(acc)
                acc_avg = np.mean(accs)
                logger.info(f'Dataset {pl.i(dataset_name)} macro-avg acc: {pl.i(acc_avg*100)}')
            else:
                raise NotImplementedError('Non personalized test')
    run()
