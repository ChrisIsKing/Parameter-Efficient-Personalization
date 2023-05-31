import os
import json
import numpy as np
import pandas as pd
import torch
import peft_u.util.models as model_util
from os.path import join as os_join
from typing import Tuple, List, Dict, Union
from logging import Logger
from argparse import ArgumentParser
from functools import partial
from torch.utils.data import DataLoader
from peft import (
    LoraConfig, 
    TaskType, 
    PrefixTuningConfig, 
    get_peft_model, 
    PeftConfig, 
    PeftModel,
    PromptTuningInit, 
    PromptTuningConfig,
    PromptEncoderConfig,
)
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from stefutil import *
from peft_u.util import *
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
    train_parser.add_argument("--leakage", type=bool, required=False, default=True)
    train_parser.add_argument("--method", type=str, required=False, default="lora", choices=["lora", "prefix", "p_tuning", "prompt_tuning"])
    train_parser.add_argument("--batch_size", type=int, required=False, default=8)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--seed", type=int, required=False, default=42)
    train_parser.add_argument("--device", type=str, required=False, default="cuda")
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    train_parser.add_argument('--personalize', type=bool, required=False, default=True)

    test_parser.add_argument("--model", type=str, required=False, default=default_md_nm)
    test_parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    test_parser.add_argument("--leakage", type=str, required=False, default=True)
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
    if logger_fl:
        logger_fl.info('Loading Peft model...')

    if peft_method == 'lora':
        peft_config: PeftConfig = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
    elif peft_method == 'prompt_tuning':
        peft_config: PeftConfig = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            num_virtual_tokens=20,
            prompt_tuning_init=PromptTuningInit.RANDOM
        )
    elif peft_method == 'p_tuning':
        peft_config: PeftConfig = PromptEncoderConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            num_virtual_tokens=20,
            encoder_hidden_size=128
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
    if logger_fl:
        logger_fl.info(f'Moving model to {device}...')
    model.to(device)
    return model, tokenizer


def smart_batching_collate(batch, tokenizer):
    """
    Collate function for PyTorch DataLoader
    """
    inputs = [example.process_template() for example in batch]
    targets = [example.process_target() for example in batch]
    batch_encoding = tokenizer(inputs, truncation=True, padding='max_length', return_tensors='pt')
    labels = tokenizer(targets, truncation=True, padding='max_length', return_tensors='pt')
    labels = labels['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    batch_encoding['labels'] = labels
    return batch_encoding


def _get_dataset_sizes(dataset: InputEgDataset):
    tr, vl, ts = dataset.train, dataset.val, dataset.test
    return dict(train_sz=len(tr), val_sz=len(vl), test_sz=len(ts))


def train_single(
        model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: InputEgDataset,
        device: str = 'cuda', seed: int = 42,
        batch_size: int = 8, num_epochs: int = 3, learning_rate: float = 2e-5, weight_decay: float = 0.01,
        output_path: str = None, user_id: str = None, verbose: bool = False
):
    output_path = os_join(output_path, f'User-{user_id}')

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

    logger_fl = get_logger(f'PEFT Train User-{user_id}', kind='file-write', file_path=os_join(output_path, 'train.log'))
    tb_writer = SummaryWriter(os_join(output_path, f'tensorboard'))
    d_log = _get_dataset_sizes(dataset)
    # if verbose:
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
                    # by empirical observation, the forward-pass tensors seems to repeat the single prediction label
                    #  => an easy solution is just consider the first word as prediction;
                    #  TODO: better & still fast ways?
                    pred = pred.split(' ')[0].strip()
                    labels = [lb.strip() for lb in true.process_target().split(', ')]
                    if pred in labels:
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
            if verbose:
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
        batch_size: int = 8, tqdm_desc: str = None, user_id: str = None, logger_fl: Logger = None,
) -> Tuple[pd.DataFrame, float]:
    label_options = sconfig(f'datasets.{dataset_name}.labels')
    lb2id = {lb: i for i, lb in enumerate(label_options)}  # sanity check each pred and true label is in config
    n_sample = len(dataset)

    collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
    ts_dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    trues, preds = np.empty(n_sample, dtype=int), np.empty(n_sample, dtype=int)
    it = tqdm(ts_dl, desc=tqdm_desc or 'Testing ')
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
            # being lenient here by dropping trailing full stop
            decoded = [d.strip().removesuffix('.').removesuffix("'") for d in decoded.split(',')]

            dec_not_in_lb = [dec for dec in decoded if dec not in label_options]
            if len(dec_not_in_lb) > 0:
                logger.warning(f'User {pl.i(user_id)} Predicted label(s) {pl.i(dec_not_in_lb)} not in label options')
                if logger_fl:
                    logger_fl.warning(f'User {user_id} Predicted label(s) {dec_not_in_lb} not in label options')

                decoded = [dec for dec in decoded if dec in label_options]

                if len(decoded) == 0:  # doesn't generate anything in the label options, declare prediction wrong
                    trues[i_sample] = lb2id[labels[0]]
                    preds[i_sample] = -1
                    continue

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
            args = dict(
                labels=[-1, *idx_lbs], target_names=['Label not in dataset', *label_options],
                zero_division=0, output_dict=True
            )
            ret = df, acc = eval_array2report_df(labels=trues, preds=preds, report_args=args, pretty=False)
            acc_str = f'{acc*100:.1f}'
            d_it['cls_acc'] = pl.i(acc_str)
            it.set_postfix(d_it)
    return ret


def _get_dataset_and_users_it(
        dataset_name: str, leakage: bool = False, uid_start_from: Union[str, int] = None
) -> Tuple[Dict[str, InputEgDataset], List[str]]:
    # from peft_u._dset_uid_too_small import uid_too_small

    dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage)

    filt = None
    # if dataset_name in uid_too_small:
    #     lst_filt = uid_too_small[dataset_name]
    #
    #     def filt(x):
    #         return x not in lst_filt
    it = iter_users(dset, start_from=uid_start_from, filter_fn=filt)
    return dset, it


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
            logger_fl = get_logger('PEFT Train fl', kind='file-write', file_path=os_join(output_path, 'train.log'))
            logger.info(f'Training PEFT w/ {pl.i(d_log)}...')
            logger_fl.info(f'Training PEFT w/ {d_log}...')

            # strt = 47  # goemotion
            # strt = 28  # hatexplain
            # strt = 127  # measuringhatespeech
            strt = None
            dset, it = _get_dataset_and_users_it(dataset_name=dataset_name, leakage=leakage, uid_start_from=strt)
            md_load_args = dict(peft_method=method, device=device, logger_fl=logger_fl)

            tm = Timer()
            if personalize:
                # global_tqdm = True
                global_tqdm = False

                n_user = len(it)
                logger.info(f'Training on users {pl.i(it)}... ')
                logger_fl.info(f'Training on users {it}... ')
                if global_tqdm:
                    it = tqdm(it, desc=f'Training on {pl.i(dataset_name)}')

                for i, uid in enumerate(it, start=1):
                    if global_tqdm:
                        d_log = dict(user=uid) | _get_dataset_sizes(dset[uid])
                        it.set_postfix({k: pl.i(v) for k, v in d_log.items()})
                    else:
                        user_ordinal = f'{pl.i(i)}/{pl.i(n_user)}'
                        logger.info(f'Launching {pl.i(dataset_name)} personalized training '
                                    f'for User {pl.i(uid)}({user_ordinal})...')
                    tm_ = Timer()

                    # # if any dataset split is empty, skip
                    # # TODO: those users should be deterministic by each dataset processing
                    # split_sizes = _get_dataset_sizes(dset[uid])
                    # if any(v == 0 for v in split_sizes.values()):
                    #     logger.info(f'Skipping User {pl.i(uid)} due to empty split w/ {pl.i(split_sizes)}...')
                    #     continue

                    # reload model for each user
                    model, tokenizer = load_model_n_tokenizer(model_name_or_path, **md_load_args)
                    train_single(
                        model=model, tokenizer=tokenizer, dataset=dset[uid], device=device, seed=seed,
                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                        weight_decay=weight_decay, output_path=output_path, user_id=uid
                    )
                    t_e_ = tm_.end()
                    if not global_tqdm:
                        logger.info(f'Training for User {pl.i(uid)} done in {pl.i(t_e_)}')
                    logger_fl.info(f'Training for User {uid} done in {t_e_}')
            else:
                raise NotImplementedError('Non personalized train')
            t_e = tm.end()
            logger.info(f'Training done in {pl.i(t_e)}')
            logger_fl.info(f'Training done in {t_e}')
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
            logger_fl = get_logger('PEFT Test fl', kind='file-write', file_path=os_join(eval_output_path, 'test.log'))
            logger_fl.info(f'Testing PEFT w/ {d_log}...')

            tm = Timer()
            if personalize:
                dset, it = _get_dataset_and_users_it(dataset_name=dataset_name, leakage=leakage)
                n_user = len(it)
                model_name_or_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_name_or_path)

                logger.info(f'Testing on users {pl.i(it)}... ')
                logger_fl.info(f'Testing on users {it}... ')

                accs = dict()
                for i, uid in enumerate(it, start=1):
                    ts = ListDataset(dset[uid].test)
                    # logger.info(f'Testing personalized PEFT for User {pl.i(uid)} w/ test split size {pl.i(len(ts))}...')
                    user_ordinal = f'{pl.i(i)}/{pl.i(n_user)}'
                    desc = f'{pl.i(now())} Testing on User {pl.i(uid)}({user_ordinal})'

                    user_str = f'User-{uid}'
                    path = os_join(model_name_or_path, user_str, 'trained')
                    assert os.path.exists(path)  # sanity check
                    # if not os.path.exists(path) or len(ts) == 0:
                    #     # TODO: see issue in training, empty split sizes
                    #     logger.info(f'Skipping User {pl.i(uid)} due to missing trained model or empty test set...')
                    #     continue

                    model, tokenizer = load_trained(model_name_or_path=path)

                    df, acc = test_single(
                        model=model, tokenizer=tokenizer, dataset=ts, batch_size=bsz,
                        dataset_name=dataset_name, tqdm_desc=desc, user_id=uid, logger_fl=logger_fl
                    )
                    path = os_join(eval_output_path, f'{user_str}.csv')
                    df.to_csv(path)

                    accs[uid] = acc
                acc_avg = np.mean(list(accs.values()))
                acc_avg_str = f'{acc_avg*100:.1f}'
                logger.info(f'Dataset {pl.i(dataset_name)} macro-avg acc: {pl.i(acc_avg_str)}')
                logger_fl.info(f'Dataset {dataset_name} macro-avg acc: {acc_avg_str}')
                with open(os_join(eval_output_path, 'accuracies.json'), 'w') as f:
                    json.dump(accs, f, indent=4)
            else:
                raise NotImplementedError('Non personalized test')
            t_e = tm.end()
            logger.info(f'Testing done in {pl.i(t_e)}')
    run()
