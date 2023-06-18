"""
Try adapters from `adapter-transformers`

Note that since `adapter-transformers` is a direct fork on HF `transformers`
and we use a different `transformers` version, make sure to set up a separate environment for `peft_u` and `adapter`

Note: remember to remove `transformers` and keep only `adapter-transformers`
"""

import math
import os
import json
from os.path import join as os_join
from typing import Tuple
from logging import Logger
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
import datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5TokenizerFast
from transformers import HoulsbyConfig, IA3Config
from transformers.adapters import T5AdapterModel
from transformers import TrainingArguments, SchedulerType
from transformers.training_args import OptimizerNames
from datasets import Dataset, DatasetDict

from stefutil import *
from peft_u.util import *
import peft_u.util.models as model_util
import peft_u.util.train as train_util
from peft_u.preprocess.load_dataset import *


logger = get_logger('Adapter Baseline')


HF_MODEL_NAME = 'google/flan-t5-base'
ADAPTER_METHODS = ['Houlsby', 'IA3']
DEFAULT_ADAPTER_METHOD = 'Houlsby'


def reduce_hf_logging():
    transformers.logging.set_verbosity_warning()
    # Disables logs for 1) training & eval info, 2) saving adapter config, 3) saving adapter params
    logger_nms = ['trainer', 'configuration_utils', 'adapters.loading']
    for logger_nm in logger_nms:
        logger_ = transformers.logging.get_logger(f'transformers.{logger_nm}')
        logger_.setLevel(transformers.logging.WARN)
    logger_ = transformers.logging.get_logger('transformers.modeling_utils')
    logger_.setLevel(transformers.logging.ERROR)  # Disables checkpoint not loading `lm_head.weight` warning

    # Disables warnings for 1) finding cached dataset, 2) saving & loading cached dataset
    logger_nms = ['builder', 'arrow_dataset']
    for logger_nm in logger_nms:
        logger_ = datasets.logging.get_logger(f'datasets.{logger_nm}')
        logger_.setLevel(datasets.logging.ERROR)


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")

    train_parser.add_argument("--model", type=str, required=False, default=HF_MODEL_NAME)
    # train_parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    # train_parser.add_argument("--leakage", type=bool, required=False, default=True)
    train_parser.add_argument(
        "--method", type=str, required=False, default=DEFAULT_ADAPTER_METHOD, choices=ADAPTER_METHODS)
    train_parser.add_argument("--batch_size", type=int, required=False, default=8)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--seed", type=int, required=False, default=42)
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    # Run on `cuda` if available, always personalize

    test_parser.add_argument("--model", type=str, required=False, default=HF_MODEL_NAME)
    # test_parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    # test_parser.add_argument("--leakage", type=str, required=False, default=True)
    test_parser.add_argument("--batch_size", type=int, required=False, default=8)

    return parser.parse_args()


def get_tokenizer(model_name_or_path: str = HF_MODEL_NAME) -> T5TokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = 512
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True  # disables warning
    return tokenizer


def load_t5_model_with_lm_head(
        model_name_or_path: str = HF_MODEL_NAME, dummy_hf_model_name: str = HF_MODEL_NAME
) -> T5AdapterModel:
    model = T5AdapterModel.from_pretrained(model_name_or_path)  # Should observe a warning on `lm_head.weight` not used

    # Use a different name so that the LM head is frozen & will not be saved to disk
    model.add_seq2seq_lm_head(head_name='__LM-Head-Frozen__')

    # Since there's not a LM head version of `T5AdapterModel`,
    # use the LM head from the HF model and set it to frozen, i.e. override `train_adapter` on the LM head
    model_dummy = AutoModelForSeq2SeqLM.from_pretrained(dummy_hf_model_name)
    state_d = model_dummy.lm_head.state_dict()
    assert set(state_d.keys()) == {'weight'}  # sanity check
    pretrained_lm_weight = state_d['weight']
    lm_head = model.heads[model.active_head]
    assert len(lm_head._modules) == 1  # sanity check, should only contain `output_embeddings`
    assert pretrained_lm_weight.shape == lm_head.get_output_embeddings().weight.shape
    lm_head_weight = lm_head.get_output_embeddings().weight
    lm_head_weight.requires_grad = False
    lm_head_weight[:] = pretrained_lm_weight
    del model_dummy, state_d
    return model


def load_adapter_model(
        model_name_or_path: str = HF_MODEL_NAME, adapter_method: str = None, user_id: str = None, verbose: bool = False,
        logger_fl: Logger = None
) -> T5AdapterModel:
    model = load_t5_model_with_lm_head(model_name_or_path=model_name_or_path, dummy_hf_model_name=model_name_or_path)
    if adapter_method == 'Houlsby':
        adapter_config = HoulsbyConfig()
    else:
        assert adapter_method == 'IA3'
        adapter_config = IA3Config()
    model.add_adapter(adapter_name=user_id, config=adapter_config)
    model.train_adapter([user_id])  # activate for training

    model_meta = get_model_meta(model)
    if verbose:
        logger.info(f'Model info: {pl.i(model_meta)}')
    if logger_fl is not None:
        logger_fl.info(f'Model info: {model_meta}')
    return model


def load_trained(model_name_or_path: str = None, user_id: str = None) -> T5AdapterModel:
    model = load_t5_model_with_lm_head()

    model.load_adapter(adapter_name_or_path=model_name_or_path)
    model.set_active_adapters([user_id])  # matching the adapter name in `load_adapter_model()`
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


if __name__ == '__main__':
    dataset_name = 'tweeteval'  # TODO: developing

    check_on_adapter()
    reduce_hf_logging()

    INSTR = "Please review the following text and indicate if it has the presence of hate speech. "\
            "Respond 'Hateful' if the text contains hate speech "\
            "and 'Non-hateful' if the text does not contain hate speech."

    def load_dset(tokenizer: T5TokenizerFast, tokenize: bool = True, **kwargs):
        fnm = 'user_data_leaked'
        with open(os_join(u.proj_path, u.dset_dir, dataset_name, f'{fnm}.json'), 'r') as f:
            data: PersonalizedDataset = json.load(f)

        def get_gen(user_data=None, split: str = None):
            def gen():
                for sid, sample in user_data[split].items():
                    txt, lb = sample['text'], sample['label']
                    assert len(lb) == 1  # single label
                    yield dict(text=txt, label=lb[0])
            return gen
        dsets = {
            uid: DatasetDict(
                train=Dataset.from_generator(get_gen(u_data, 'train')),
                validation=Dataset.from_generator(get_gen(u_data, 'val')),
                test=Dataset.from_generator(get_gen(u_data, 'test'))
            ) for uid, u_data in data.items()
        }

        if tokenize:
            def map_text(txt: str):
                return f'{INSTR} Text: {txt} Label: '

            def map_single(batch):
                inputs = [map_text(txt) for txt in batch['text']]
                labels = batch['label']
                tok_args = dict(
                    truncation=True, padding='max_length',
                    return_tensors='pt'
                )
                ret = tokenizer(inputs, **tok_args)
                labels = tokenizer(labels, **tok_args)['input_ids']
                labels[labels == tokenizer.pad_token_id] = -100  # `-100` is ignored in loss
                ret['labels'] = labels
                return ret
            return {uid: u_dset.map(map_single, batched=True, **kwargs) for uid, u_dset in dsets.items()}
        else:
            return dsets

    def command_prompt():
        args = parse_args()
        cmd = args.mode
        if cmd == 'train':
            model_name_or_path, method = args.model, args.method
            # dataset_name, leakage = args.dataset_name, args.leakage
            batch_size, num_epochs, learning_rate = args.batch_size, args.num_epochs, args.learning_rate
            weight_decay = args.weight_decay
            seed = args.seed
            output_dir = args.output_dir

            get_args = dict(model_name=model_name_or_path, name=output_dir, method=method, method_key='adapter')
            output_path = model_util.get_train_output_path(**get_args, dataset_name=dataset_name)
            d_log = dict(
                model_name_or_path=model_name_or_path, method=method,
                # dataset_name=dataset_name, leakage=leakage,
                batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                seed=seed,
                output_dir=output_dir, output_path=output_path
            )
            fnm = os_join(output_path, f'train_{now(for_path=True)}.log')
            logger_fl = get_logger('Adapter Train fl', kind='file-write', file_path=fnm)
            logger.info(f'Training Adapter w/ {pl.i(d_log)}...')
            logger_fl.info(f'Training Adapter w/ {d_log}...')

            tokenizer = get_tokenizer(model_name_or_path=model_name_or_path)
            dsets = load_dset(tokenizer=tokenizer, remove_columns=['text', 'label'])

            def get_train_meta(user_id: str = None, verbose: bool = False) -> Tuple[TrainingArguments, Logger, str]:
                output_path_ = os_join(output_path, uid2u_str(user_id))
                train_args = dict(
                    output_dir=output_path_,
                    do_train=True,
                    do_eval=True,
                    optim=OptimizerNames.ADAMW_TORCH,
                    learning_rate=learning_rate,
                    lr_scheduler_type=SchedulerType.LINEAR,  # same w/ adapter baseline
                    warmup_ratio=1e-2,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=num_epochs,
                    weight_decay=weight_decay,
                    remove_unused_columns=False,
                    disable_tqdm=True,
                    log_level='warning',
                    logging_strategy='steps',
                    logging_steps=1,
                    evaluation_strategy='epoch',
                    save_strategy='epoch',
                    report_to='none',
                    load_best_model_at_end=True,
                    metric_for_best_model='eval_loss',
                    greater_is_better=False
                )
                if verbose:
                    logger.info(f'Training args: {pl.fmt(train_args)}')
                train_args = TrainingArguments(**train_args)

                fnm_ = os_join(output_path_, f'train_user-{user_id}.log')
                logger_fl_ = get_logger(f'Adapter user-{user_id} Train fl', kind='file-write', file_path=fnm_)
                logger_fl_.info(f'Training args: {pl.id(train_args.to_dict())}')
                return train_args, logger_fl_, output_path_

            tm = Timer()
            it = iter_users(dataset=dsets)[:4]  # TODO: debugging
            n_user = len(it)
            logger.info(f'Training on users {pl.i(it)}... ')
            logger_fl.info(f'Training on users {it}... ')

            for i, uid in enumerate(it, start=1):
                dset = dsets[uid]
                model = load_adapter_model(
                    model_name_or_path=model_name_or_path, adapter_method=method, user_id=uid, logger_fl=logger_fl
                )
                args, logger_fl__, output_path__ = get_train_meta(user_id=uid)
                trainer = train_util.MyAdapterTrainer(
                    logger_fl=logger_fl__, model=model, args=args, tokenizer=tokenizer,
                    train_dataset=dset['train'], eval_dataset=dset['validation']
                )

                user_str_ordinal = train_util.get_user_str_w_ordinal(user_id=uid, user_idx=i, n_user=n_user)
                logger.info(f'Launching {pl.i(dataset_name)} personalized training for User {user_str_ordinal}...')

                transformers.set_seed(seed)
                tm_ = Timer()
                trainer.train()
                t_e_ = tm_.end()
                logger.info(f'Training for User {pl.i(uid)} done in {pl.i(t_e_)}')
                logger_fl.info(f'Training for User {uid} done in {t_e_}')
                model.save_adapter(save_directory=os_join(output_path__, 'trained'), adapter_name=uid)
            t_e = tm.end()
            logger.info(f'Training done in {pl.i(t_e)}')
            logger_fl.info(f'Training done in {t_e}')
        else:
            assert cmd == 'test'
            model_name_or_path = args.model
            # dataset_name, leakage = args.dataset_name, args.leakage
            bsz = args.batch_size

            date = now(fmt='short-date')
            eval_output_path = os_join(u.proj_path, 'eval', f'{model_name_or_path}_Eval-{date}')
            os.makedirs(eval_output_path, exist_ok=True)

            d_log = dict(
                model_name_or_path=model_name_or_path,
                # dataset_name=dataset_name, leakage=leakage,
                batch_size=bsz
            )
            logger.info(f'Testing Adapter w/ {pl.i(d_log)}...')
            fnm = os_join(eval_output_path, f'test_{now(for_path=True)}.log')
            logger_fl = get_logger('Adapter Test fl', kind='file-write', file_path=fnm)
            logger_fl.info(f'Testing Adapter w/ {d_log}...')

            model_name_or_path = model_util.prepend_local_model_path(model_path=model_name_or_path)
            tokenizer = get_tokenizer(model_name_or_path=HF_MODEL_NAME)
            dsets = load_dset(tokenizer=tokenizer)

            tm_ = Timer()

            label_options = sconfig(f'datasets.{dataset_name}.labels')
            lb2id = {lb: i for i, lb in enumerate(label_options)}  # sanity check each pred and true label is in config

            it = iter_users(dataset=dsets)[:4]  # TODO: debugging
            n_user = len(it)
            logger.info(f'Testing on users {pl.i(it)}... ')
            logger_fl.info(f'Testing on users {it}... ')

            accs = dict()
            for i, uid in enumerate(it, start=1):
                dset = dsets[uid]['test']

                user_str = uid2u_str(uid)  # load trained model for each user
                path = os_join(model_name_or_path, user_str, 'trained')
                assert os.path.exists(path)  # sanity check
                model = load_trained(model_name_or_path=path, user_id=uid)

                n_sample = len(dset)
                trues, preds = np.empty(n_sample, dtype=int), np.empty(n_sample, dtype=int)

                n_it = math.ceil(n_sample / 8)
                user_args = dict(user_id=uid, user_idx=i, n_user=n_user)
                it = train_util.get_user_test_pbar(it=group_n(range(n_sample), n=bsz), **user_args, total=n_it)
                d_it = dict(dataset_size=pl.i(n_sample))
                it.set_postfix(d_it)

                for i_ba, idxs in enumerate(it):
                    idxs = [int(idx) for idx in idxs]
                    inputs = {k: torch.tensor(v) for k, v in dset[idxs].items() if k not in ['text', 'label', 'labels']}
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    labels = dset[idxs]['label']

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=16)
                    lst_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    for idx, lb, dec in zip(idxs, labels, lst_decoded):
                        if lb == dec:
                            preds[idx] = trues[idx] = lb2id[lb]
                        else:
                            preds[idx] = lb2id[dec] if dec in lb2id else -1
                            trues[idx] = lb2id[lb]

                    if i_ba + 1 == n_it:  # last iteration
                        accs[uid] = train_util.test_user_update_postfix_n_write_df(
                            label_options=label_options, trues=trues, preds=preds, pbar=it, d_postfix=d_it,
                            df_out_path=os_join(eval_output_path, f'{user_str}.csv')
                        )
            out_args = dict(d_accs=accs, logger_fl=logger_fl, eval_output_path=eval_output_path)
            train_util.log_n_save_test_results(dataset_name=dataset_name, **out_args)

            t_e_ = tm_.end()
            logger.info(f'Testing done in {pl.i(t_e_)}')
            logger_fl.info(f'Testing done in {t_e_}')
    command_prompt()
