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
from argparse import ArgumentParser

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5TokenizerFast
from transformers import HoulsbyConfig, IA3Config
from transformers.adapters import T5AdapterModel
from transformers import AdapterTrainer, TrainingArguments
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from stefutil import *


logger = get_logger('Adapter Baseline')


_DIRS = __file__.split(os.sep)
BASE_PATH, PROJ_DIR, PKG_DIR = os.sep.join(_DIRS[:-3]), _DIRS[-3], _DIRS[-2]
DSET_DIR = 'data'


HF_MODEL_NAME = 'google/flan-t5-base'
ADAPTER_METHODS = ['Houlsby', 'IA3']
DEFAULT_ADAPTER_METHOD = 'Houlsby'


def check_on_adapter():
    try:
        d_log = dict(transformers_version=transformers.__version__, adapter_version=transformers.adapters.__version__)
        logger.info(pl.i(d_log))
    except AttributeError:
        raise ImportError('This script is intended for `adapter-transformers`, '
                          'please install `adapter-transformers` instead of `transformers`')


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


def load_t5_model_with_lm_head_n_tokenizer(
        model_name_or_path: str = HF_MODEL_NAME, dummy_hf_model_name: str = HF_MODEL_NAME
) -> Tuple[T5AdapterModel, T5TokenizerFast]:
    model = T5AdapterModel.from_pretrained(model_name_or_path)  # Should observe a warning on `lm_head.weight` not used

    # Use a different name so that the LM head is not saved to disk
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

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = 512
    return model, tokenizer


if __name__ == '__main__':
    check_on_adapter()

    transformers.logging.set_verbosity_warning()
    logger_ = transformers.logging.get_logger('transformers.trainer')  # Disables training & eval info log
    logger_.setLevel(transformers.logging.WARN)

    INSTR = "Please review the following text and indicate if it has the presence of hate speech. "\
            "Respond 'Hateful' if the text contains hate speech "\
            "and 'Non-hateful' if the text does not contain hate speech."

    def load_dset(tokenizer: T5TokenizerFast, tokenize: bool = True, **kwargs):
        dnm, fnm = 'tweeteval', 'user_data_leaked'
        with open(os_join(BASE_PATH, PROJ_DIR, DSET_DIR, dnm, f'{fnm}.json'), 'r') as f:
            data = json.load(f)
        data = data['0']  # Take 1st user arbitrarily

        def get_gen(split: str):
            def gen():
                for sid, sample in data[split].items():
                    txt, lb = sample['text'], sample['label']
                    assert len(lb) == 1  # single label
                    yield dict(text=txt, label=lb[0])
            return gen
        dset = DatasetDict(
            train=Dataset.from_generator(get_gen('train')),
            validation=Dataset.from_generator(get_gen('val')),
            test=Dataset.from_generator(get_gen('test'))
        )

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
            return dset.map(map_single, batched=True, **kwargs)
        else:
            return dset

    # DEBUG = True
    DEBUG = False
    # md_nm = 'google/flan-t5-small' if DEBUG else HF_MODEL_NAME
    DEBUG_ADAPTER_NM = 'debug'

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

            date = now(fmt='short-date')
            _md_nm = model_name_or_path
            if '/' in _md_nm:
                org, _md_nm = _md_nm.split('/')
            meta = dict(md_nm=_md_nm, adapter=method)
            output_path = os_join(BASE_PATH, PROJ_DIR, 'models', f'{date}_{pl.pa(meta)}_{output_dir}')
            os.makedirs(output_path, exist_ok=True)
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

            model, tokenizer = load_t5_model_with_lm_head_n_tokenizer(
                model_name_or_path=model_name_or_path, dummy_hf_model_name=model_name_or_path
            )
            if method == 'Houlsby':
                adapter_config = HoulsbyConfig()
            else:
                assert method == 'IA3'
                adapter_config = IA3Config()
            model.add_adapter(adapter_name=DEBUG_ADAPTER_NM, config=adapter_config)
            model.train_adapter([DEBUG_ADAPTER_NM])  # activate for training
            model_meta = get_model_meta(model)
            logger.info(f'Model info: {pl.i(model_meta)}')
            logger_fl.info(f'Model info: {model_meta}')

            dset = load_dset(tokenizer=tokenizer, remove_columns=['text', 'label'])

            train_args = TrainingArguments(
                output_dir=output_path,
                do_train=True,
                do_eval=True,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                remove_unused_columns=False,
                disable_tqdm=True,  # For my custom pbar
                evaluation_strategy='epoch'
            )
            tr, vl = dset['train'], dset['validation']
            if DEBUG:
                tr = tr.select(range(16))
                vl = vl.select(range(16))

            trainer = AdapterTrainer(
                model=model, args=train_args, tokenizer=tokenizer, train_dataset=tr, eval_dataset=vl
            )
            callbacks = trainer.callback_handler.callbacks
            trainer.callback_handler.callbacks = [  # Remove internal callback
                c for c in callbacks if str(c.__class__) != "<class 'transformers.trainer_callback.PrinterCallback'>"
            ]
            trainer.add_callback(MyProgressCallback())

            trainer.train()
            model.save_adapter(save_directory=output_path, adapter_name=DEBUG_ADAPTER_NM)
        else:
            assert cmd == 'test'
            model_name_or_path = args.model
            # dataset_name, leakage = args.dataset_name, args.leakage
            bsz = args.batch_size

            model, tokenizer = load_t5_model_with_lm_head_n_tokenizer()

            adapter_path = os_join(BASE_PATH, PROJ_DIR, 'models', model_name_or_path)
            model.load_adapter(adapter_name_or_path=adapter_path)
            model.set_active_adapters([DEBUG_ADAPTER_NM])
            model.eval()

            dset = load_dset(tokenizer=tokenizer)['test']
            idxs_gen = group_n(range(len(dset)), n=bsz)
            total = math.ceil(len(dset) / 8)

            n_total, n_correct = 0, 0
            for i_ba, idxs in enumerate(tqdm(idxs_gen, desc='Testing', total=total)):
                idxs = [int(idx) for idx in idxs]
                inputs = {k: torch.tensor(v) for k, v in dset[idxs].items() if k not in ['text', 'label', 'labels']}
                labels = dset[idxs]['label']

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=16)
                lst_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for lb, dec in zip(labels, lst_decoded):
                    n_total += 1
                    if lb == dec:
                        n_correct += 1
            mic(n_correct / n_total)
    command_prompt()
