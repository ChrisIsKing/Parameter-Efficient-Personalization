"""
Try adapters from `adapter-transformers`

Note that since `adapter-transformers` is a direct fork on HF `transformers`
and we use a different `transformers` version, make sure to set up a separate environment for `peft_u` and `adapter`

Note: remember to remove `transformers` and keep only `adapter-transformers`
"""

import math
import os
from os.path import join as os_join
from typing import Tuple, List, Dict, Any
from logging import Logger
from argparse import Namespace

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
import peft_u.trainer.train as train_util
from peft_u.preprocess.load_dataset import *
from peft_u.trainer import HF_MODEL_NAME, get_arg_parser


logger = get_logger('Adapter Baseline')


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
    # Note downloading & preparing the generator dataset is printing, doesn't go through the logger so can't disable


def parse_args():
    return get_arg_parser(default_method=DEFAULT_ADAPTER_METHOD, method_choices=ADAPTER_METHODS).parser.parse_args()


def get_tokenizer(model_name_or_path: str = HF_MODEL_NAME) -> T5TokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = 512
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True  # disables warning
    return tokenizer


def load_dataset(
        dataset_name: str = None, leakage: bool = None, seed: int = None, load_users_args: Dict[str, Any] = None,
        tokenizer: T5TokenizerFast = None, tokenize: bool = True, use_user_profile:bool = False, **kwargs
) -> Tuple[Dict[str, DatasetDict], List[str]]:
    dsets = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage, seed=seed, use_user_profile=use_user_profile)

    user_ids = iter_users(dataset=dsets, **(load_users_args or dict()))
    n_original_user, n_user = len(dsets), len(user_ids)
    if n_original_user > n_user:  # saves later processing time, matters for huge datasets
        logger.info(f'Filtering user ids: {pl.i(n_original_user)} => {pl.i(n_user)}')
        dsets = {uid: dsets[uid] for uid in user_ids}

    def get_gen(user_examples: List[InputExample]):
        def gen():
            for eg in user_examples:
                yield dict(text=eg.process_template(), label=eg.process_target())
        return gen

    c_dir = model_util.get_hf_dataset_cache_dir()
    dsets = {
        uid: DatasetDict(
            train=Dataset.from_generator(get_gen(u_data.train), cache_dir=c_dir),
            validation=Dataset.from_generator(get_gen(u_data.val), cache_dir=c_dir),
            test=Dataset.from_generator(get_gen(u_data.test), cache_dir=c_dir)
        ) for uid, u_data in dsets.items()
    }

    if tokenize:
        def map_single(batch):
            return train_util.BatchCollator(tokenizer)(texts=batch['text'], labels=batch['label'])
        return {uid: u_dset.map(map_single, batched=True, **kwargs) for uid, u_dset in dsets.items()}, user_ids
    else:
        return dsets, user_ids


def get_train_meta(
        args: Namespace = None, output_path: str = None, user_id: str = None, verbose: bool = False
) -> Tuple[TrainingArguments, Logger, str]:
    bsz = args.batch_size
    output_path = os_join(output_path, uid2u_str(user_id))
    train_args = dict(
        output_dir=output_path,
        do_train=True,
        do_eval=True,
        fp16=True,
        optim=OptimizerNames.ADAMW_TORCH,
        learning_rate=args.learning_rate,
        lr_scheduler_type=SchedulerType.LINEAR,  # same w/ adapter baseline
        warmup_ratio=1e-2,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
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

    fnm_ = os_join(output_path, f'train_user-{user_id}.log')
    logger_fl = get_logger(f'Adapter user-{user_id} Train fl', kind='file-write', file_path=fnm_)
    logger_fl.info(f'Training args: {pl.id(train_args.to_dict())}')
    return train_args, logger_fl, output_path


def load_t5_model_with_lm_head(
        model_name_or_path: str = HF_MODEL_NAME, dummy_hf_model_name: str = HF_MODEL_NAME
) -> T5AdapterModel:
    model = T5AdapterModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)  # Should observe a warning on `lm_head.weight` not used

    # Use a different name so that the LM head is frozen & will not be saved to disk
    model.add_seq2seq_lm_head(head_name='__LM-Head-Frozen__')

    # Since there's not a LM head version of `T5AdapterModel`,
    # use the LM head from the HF model and set it to frozen, i.e. override `train_adapter` on the LM head
    model_dummy = AutoModelForSeq2SeqLM.from_pretrained(dummy_hf_model_name, torch_dtype=torch.bfloat16)
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
    _same_param_exp = False
    if adapter_method == 'Houlsby':
        adapter_config = HoulsbyConfig(reduction_factor=512 if _same_param_exp else 16)
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
    check_on_adapter()
    reduce_hf_logging()

    def command_prompt():
        args = parse_args()
        cmd = args.mode
        if cmd == 'train':
            model_name_or_path, method = args.model, args.method
            dataset_name, leakage = args.dataset_name, args.leakage
            seed = args.seed
            use_user_profile = args.use_user_profile
            output_path, logger_fl = train_util.setup_train_output_path_n_loggers(args=args, approach='adapter')
            is_generative = sconfig(f'datasets.{dataset_name}.is_generative')
            leakage = leakage if not is_generative else False # force no leakage for generative

            tokenizer = get_tokenizer(model_name_or_path=model_name_or_path)

            # strt = '72'  # `hatexplain`
            # strt = 1163  # `cockamamie`
            # strt = 1936  # `wikidetox.ia3`
            # strt = 36  # `wikidetox.houlsby`
            # strt = 5  # `studemo`
            strt = None
            # end = 47  # `wikidetox`
            end = None
            dsets, it = load_dataset(
                dataset_name=dataset_name, leakage=leakage, seed=seed,
                load_users_args=dict(start_from=strt, end_at=end),
                tokenizer=tokenizer, remove_columns=['text', 'label'],
                use_user_profile=use_user_profile
            )

            tm = Timer()
            n_user = len(it)
            logger.info(f'Training on users {pl.i(it)}... ')
            logger_fl.info(f'Training on users {it}... ')

            for i, uid in enumerate(it, start=1):
                dset = dsets[uid]
                model = load_adapter_model(
                    model_name_or_path=model_name_or_path, adapter_method=method, user_id=uid, logger_fl=logger_fl
                )
                train_args, logger_fl_, output_path_ = get_train_meta(args=args, output_path=output_path, user_id=uid)
                trainer = train_util.MyAdapterTrainer(
                    logger_fl=logger_fl_, model=model, args=train_args, tokenizer=tokenizer,
                    train_dataset=dset['train'], eval_dataset=dset['validation'],
                )

                user_str_ordinal = train_util.get_user_str_w_ordinal(user_id=uid, user_idx=i, n_user=n_user)
                logger.info(f'Launching {pl.i(dataset_name)} personalized training for User {user_str_ordinal}...')
                d_log = dict(train_sz=len(dset['train']), val_sz=len(dset['validation']), test_sz=len(dset['test']))
                logger.info(f'Dataset sizes: {pl.i(d_log)}')
                logger_fl.info(f'Dataset sizes: {d_log}')

                transformers.set_seed(seed)
                tm_ = Timer()
                trainer.train()
                t_e_ = tm_.end()
                logger.info(f'Training for User {pl.i(uid)} done in {pl.i(t_e_)}')
                logger_fl.info(f'Training for User {uid} done in {t_e_}')
                model.save_adapter(save_directory=os_join(output_path_, 'trained'), adapter_name=uid)

                torch.cuda.empty_cache()
            t_e = tm.end()
            logger.info(f'Training done in {pl.i(t_e)}')
            logger_fl.info(f'Training done in {t_e}')
        else:
            assert cmd == 'test'
            model_name_or_path = args.model
            dataset_name, leakage = args.dataset_name, args.leakage
            bsz = args.batch_size
            seed = args.seed
            use_user_profile = args.use_user_profile
            is_generative = sconfig(f'datasets.{dataset_name}.is_generative')
            leakage = leakage if not is_generative else False # force no leakage for generative

            date = now(fmt='short-date')
            eval_output_path = os_join(u.proj_path, 'eval', f'{model_name_or_path}_Eval-{date}' if not use_user_profile else f'{model_name_or_path}__UserProfile-Eval-{date}')
            os.makedirs(eval_output_path, exist_ok=True)

            d_log = dict(
                model_name_or_path=model_name_or_path,
                dataset_name=dataset_name, leakage=leakage,
                batch_size=bsz
            )
            logger.info(f'Testing Adapter w/ {pl.i(d_log)}...')
            fnm = os_join(eval_output_path, f'test_{now(for_path=True)}.log')
            logger_fl = get_logger('Adapter Test fl', kind='file-write', file_path=fnm)
            logger_fl.info(f'Testing Adapter w/ {d_log}...')

            model_name_or_path = model_util.prepend_local_model_path(model_path=model_name_or_path)
            tokenizer = get_tokenizer(model_name_or_path=HF_MODEL_NAME)
            dsets, it = load_dataset(dataset_name=dataset_name, leakage=leakage, seed=seed, tokenizer=tokenizer, use_user_profile=use_user_profile)

            n_user = len(it)

            label_options = sconfig(f'datasets.{dataset_name}.labels')
            d_log = dict(users=it, label_options=label_options)
            logger.info(f'Testing w/ {pl.i(d_log)}...')
            logger_fl.info(f'Testing w/ {d_log}...')

            get_pred = train_util.GetPredId(label_options=label_options, logger_fl=logger_fl)
            accs = dict()
            tm_ = Timer()
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
                        out = get_pred(decoded=dec, labels=lb, user_id=uid)
                        preds[idx], trues[idx] = out.pred, out.true

                    if i_ba + 1 == n_it:  # last iteration
                        accs[uid] = train_util.test_user_update_postfix_n_write_df(
                            label_options=label_options, trues=trues, preds=preds, pbar=it, d_postfix=d_it,
                            df_out_path=os_join(eval_output_path, f'{user_str}.csv')
                        )
            out_args = dict(d_accs=accs, logger_fl=logger_fl, eval_output_path=eval_output_path, is_generative=is_generative)
            train_util.log_n_save_test_results(dataset_name=dataset_name, **out_args)

            t_e_ = tm_.end()
            logger.info(f'Testing done in {pl.i(t_e_)}')
            logger_fl.info(f'Testing done in {t_e_}')
    command_prompt()

    def check_learnable_param():
        # method = 'Houlsby'
        method = 'IA3'
        model = load_adapter_model(HF_MODEL_NAME, adapter_method=method, user_id='bla')
        mic(get_trainable_param_meta(model, fmt='int'))
    # check_learnable_param()

    def check_untrained_users():
        from tqdm import tqdm

        # model_dnm = '23-06-22_{adapter=Houlsby, md_nm=flan-t5-base, ds=wikidetox}'
        model_dnm = '23-06-22_{adapter=IA3, md_nm=flan-t5-base, ds=wikidetox}'
        model_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_dnm)

        dsets = load_dataset_with_prompts(dataset_name='wikidetox', leakage=True, seed=42)
        users = iter_users(dataset=dsets)

        untrained = []
        for uid in tqdm(users, desc='Looking for untrained users'):
            path = os_join(model_path, uid2u_str(uid), 'trained')
            if not os.path.exists(path):
                untrained.append(uid)
        mic(untrained)
    # check_untrained_users()

