import os
from os.path import join as os_join
from typing import Tuple, List, Dict, Union, Any
from logging import Logger
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaConfig
)
from peft import get_peft_model, PeftConfig, PeftModel, PeftModelForSeq2SeqLM
from peft import TaskType, LoraConfig,  PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from peft import PromptTuningInit, PromptEncoderReparameterizationType
from tqdm import tqdm

from stefutil import *
from peft_u.util import *
import peft_u.util.models as model_util
import peft_u.trainer.train as train_util
from peft_u.preprocess.load_dataset import *
from peft_u.trainer import HF_MODEL_NAME, get_arg_parser


logger = get_logger('PEFT Baseline')


PEFT_METHODS = ["lora", "prefix", "p_tuning", "prompt_tuning"]
DEFAULT_PEFT_METHOD = 'lora'


def parse_args():
    out = get_arg_parser(default_method=DEFAULT_PEFT_METHOD, method_choices=PEFT_METHODS)
    out.test_parser.add_argument("--zeroshot", type=bool, required=False, default=False)
    out.test_parser.add_argument("--use_user_profile", type=lambda x: (str(x).lower() == 'true'), required=False, default=False)
    return out.parser.parse_args()


def load_model(
        model_name_or_path: str = HF_MODEL_NAME, peft_method: str = DEFAULT_PEFT_METHOD,
        verbose: bool = False, logger_fl: Logger = None, is_generative: bool = False
) -> PeftModelForSeq2SeqLM:
    log = model_util.LoadModelLogging(logger=logger, logger_fl=logger_fl, verbose=verbose)
    cache_dir = log.get_n_log_cache_dir(model_name_or_path=model_name_or_path)

    if 'llama' in model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_name_or_path)
        if not is_generative:
            config.num_labels = 2
            model = LlamaForSequenceClassification.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, torch_dtype=torch.bfloat16) #TODO fp16)
        else:
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16) #TODO fp16)

    if peft_method is not None:
        ca.check_mismatch('PEFT method', peft_method, PEFT_METHODS)
        if verbose:
            logger.info('Loading Peft model...')
        if logger_fl:
            logger_fl.info('Loading Peft model...')

        config_d: Dict[str, Any] = dict(task_type=(TaskType.CAUSAL_LM if 'llama' in model_name_or_path.lower() else TaskType.SEQ_2_SEQ_LM), inference_mode=False)
        _same_param_exp = False
        if peft_method == 'lora':
            peft_config = LoraConfig(**config_d, r=1 if _same_param_exp else 8, lora_alpha=32, lora_dropout=0.1)
        else:
            _encoder_insertion_only = True  # for prompting in encoder input only
            config_d['num_virtual_tokens'] = 20
            if peft_method == 'prefix':
                if _same_param_exp:
                    config_d['num_virtual_tokens'] = 6
                peft_config = PrefixTuningConfig(**config_d)
            elif peft_method == 'p_tuning':
                if _encoder_insertion_only:
                    if _same_param_exp:
                        config_d.update(dict(encoder_hidden_size=12, encoder_num_layers=1))
                    else:
                        config_d.update(dict(encoder_hidden_size=128, encoder_num_layers=2))
                    peft_config: PeftConfig = PromptEncoderConfig(
                        **config_d, num_transformer_submodules=1,
                        # the original paper uses LSTM
                        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM
                    )
                else:  # old experiments used this setup
                    assert not _same_param_exp  # experiment intended for updated p-tuning only
                    peft_config: PeftConfig = PromptEncoderConfig(
                        **config_d, num_transformer_submodules=2,
                        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP
                    )
            else:
                assert peft_method == 'prompt_tuning'
                if _same_param_exp:
                    config_d['num_virtual_tokens'] = 144
                if not _encoder_insertion_only:
                    assert not _same_param_exp  # experiment intended for updated p-tuning only
                peft_config: PeftConfig = PromptTuningConfig(
                    **config_d, prompt_tuning_init=PromptTuningInit.RANDOM,
                    num_transformer_submodules=1 if _encoder_insertion_only else 2
                )
        model = get_peft_model(model, peft_config)

    log.get_n_log_model_meta(model=model)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load_trained(model_name_or_path: str = None, verbose: bool = False) -> Tuple[PeftModel, PreTrainedTokenizer]:
    cache_dir = model_util.get_hf_model_cache_dir()
    if verbose:
        logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')

    config = PeftConfig.from_pretrained(model_name_or_path)

    if 'llama' in model_name_or_path.lower():
        model = LlamaForSequenceClassification.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16) #TODO fp16)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        # model.resize_token_embeddings(len(tokenizer))
        tokenizer.padding_side = "left"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16) #TODO fp16)
        model = PeftModel.from_pretrained(model, model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


class TrainSaver:
    def __init__(self, model: PeftModel, output_base_path: str = None, verbose: bool = False):
        self.model = model
        self.output_base_path = output_base_path
        self.verbose = verbose

    def __call__(self, output_dir_nm: str):
        out = os_join(self.output_base_path, output_dir_nm)
        self.model.save_pretrained(out)
        if self.verbose:
            logger.info(f'Model saved to {pl.i(out)}')


def _get_dataset_and_users_it(
        dataset_name: str, leakage: bool = False,
        uid_start_from: Union[str, int] = None, uid_end_at: Union[str, int] = None, seed: int = None, use_user_profile: bool = False, is_generative: bool = False
) -> Tuple[Dict[str, InputEgDataset], List[str]]:
    # from peft_u._dset_uid_too_small import uid_too_small

    dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage, seed=seed, use_user_profile=use_user_profile)

    filt = None
    # if dataset_name in uid_too_small:
    #     lst_filt = uid_too_small[dataset_name]
    #
    #     def filt(x):
    #         return x not in lst_filt
    it = iter_users(dataset=dset, start_from=uid_start_from, end_at=uid_end_at, filter_fn=filt)
    return dset, it


if __name__ == '__main__':
    check_not_on_adapter()

    def command_prompt():
        args = parse_args()
        cmd = args.mode
        if cmd == 'train':
            model_name_or_path, method = args.model, args.method
            dataset_name, leakage = args.dataset_name, args.leakage
            seed = args.seed
            use_user_profile = args.use_user_profile
            output_path, logger_fl = train_util.setup_train_output_path_n_loggers(args=args, approach='peft')
            is_generative = sconfig(f'datasets.{dataset_name}.is_generative')
            leakage = leakage if not is_generative else False # force no leakage for generative
            # strt = 23  # goemotion
            # strt = 28  # hatexplain
            # strt = 5021  # `measuringhatespeech.lora`
            # strt = 3896  # `measuringhatespeech.prefix`
            # strt = 3342  # `measuringhatespeech.p_tuning`
            # strt = 2450  # `measuringhatespeech.prompt_tuning`
            # strt = 1705   # `cockamamie.p_tuning`
            # strt = 1731  # `cockamamie.prompt_tuning`
            # strt = 366  # `wikidetox`
            # strt = 2687  # wikidetox.p_tuning`
            # strt = '44590228'  # `unhealthyconversations`
            strt = None
            load_args = dict(dataset_name=dataset_name, leakage=leakage, seed=seed)
            dset, it = _get_dataset_and_users_it(**load_args, uid_start_from=strt, use_user_profile=use_user_profile, is_generative=is_generative)

            tm = Timer()
            # global_tqdm = True
            global_tqdm = False

            n_user = len(it)
            logger.info(f'Training on users {pl.i(it)}... ')
            logger_fl.info(f'Training on users {it}... ')
            if global_tqdm:
                it = tqdm(it, desc=f'Training on {pl.i(dataset_name)}')

            tokenizer = train_util.load_tokenizer()
            trainer = train_util.MyTrainer(
                tokenizer=tokenizer,
                seed=seed, batch_size=args.batch_size, num_epochs=args.num_epochs,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                output_path=output_path, saver_cls=TrainSaver
            )
            for i, uid in enumerate(it, start=1):
                if global_tqdm:
                    d_log = dict(user=uid) | get_dataset_sizes(dset[uid])
                    it.set_postfix({k: pl.i(v) for k, v in d_log.items()})
                else:
                    user_str_ordinal = train_util.get_user_str_w_ordinal(user_id=uid, user_idx=i, n_user=n_user)
                    logger.info(f'Launching {pl.i(dataset_name)} personalized training for User {user_str_ordinal}...')
                tm_ = Timer()

                # if any dataset split is empty, skip
                split_sizes = get_dataset_sizes(dset[uid])
                if any(v == 0 for v in split_sizes.values()):
                    logger.info(f'Skipping User {pl.i(uid)} due to empty split w/ {pl.i(split_sizes)}...')
                    continue

                model = load_model(model_name_or_path=model_name_or_path, peft_method=method, logger_fl=logger_fl)
                model = model.to(torch.bfloat16)
                model.config.pad_token_id = tokenizer.pad_token_id

                trainer(model=model, dataset=dset[uid], user_id=uid, save_per_epoch=False, is_generative=is_generative)
                t_e_ = tm_.end()
                if not global_tqdm:
                    logger.info(f'Training for User {pl.i(uid)} done in {pl.i(t_e_)}')
                logger_fl.info(f'Training for User {uid} done in {t_e_}')
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
            zeroshot = args.zeroshot
            is_generative = sconfig(f'datasets.{dataset_name}.is_generative')
            leakage = leakage if not is_generative else False # force no leakage for generative

            date = now(fmt='short-date')
            if zeroshot:
                d = dict(md=model_util.hf_model_name_drop_org(model_name_or_path), dnm=dataset_name)
                eval_out_str = f'{date}_ZeroShot-Eval_{pl.pa(d)}' if not use_user_profile else f'{date}_UserProfle-ZeroShot-Eval_{pl.pa(d)}'
            else:
                eval_out_str = f'{model_name_or_path}_Eval-{date}' if not use_user_profile else f'{model_name_or_path}_UserProfle-Eval-{date}'
            eval_output_path = os_join(u.eval_path, eval_out_str)
            os.makedirs(eval_output_path, exist_ok=True)

            d_log = dict(
                model_name_or_path=model_name_or_path, dataset_name=dataset_name, leakage=leakage,
                batch_size=bsz, zeroshot=zeroshot
            )
            logger.info(f'Testing PEFT w/ {pl.i(d_log)}...')
            fnm = os_join(eval_output_path, f'test_{now(for_path=True)}.log')
            logger_fl = get_logger('PEFT Test fl', kind='file-write', file_path=fnm)
            logger_fl.info(f'Testing PEFT w/ {d_log}...')

            tm = Timer()
            model = None
            tokenizer = train_util.load_tokenizer(model_name_or_path)
            if zeroshot:  # Load global model for all users
                load_args = dict(peft_method=None, verbose=True, logger_fl=logger_fl)
                model = load_model(model_name_or_path=model_name_or_path, is_generative=is_generative, **load_args)
                model.config.pad_token_id = tokenizer.pad_token_id
                model.eval()
            else:
                model_name_or_path = model_util.prepend_local_model_path(model_path=model_name_or_path)
            if type(model).__name__ in ('LlamaForSequenceClassification','LlamaForCausalLM'):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
                model.resize_token_embeddings(len(tokenizer))
                # model.generation_config.pad_token_id = tokenizer.pad_token_id
            # strt = 29044976  # unhealthyconversations
            strt = None
            load_args = dict(dataset_name=dataset_name, leakage=leakage, seed=seed, use_user_profile=use_user_profile)
            dset, it = _get_dataset_and_users_it(**load_args, uid_start_from=strt, is_generative=is_generative)
            n_user = len(it)
            d_log = dict(users=it)#, label_options=sconfig(f'datasets.{dataset_name}.labels'))
            logger.info(f'Testing w/ {pl.i(d_log)}...')
            logger_fl.info(f'Testing w/ {d_log}...')

            import gc
            accs = dict()
            tester = train_util.MyTester(
                tokenizer=tokenizer, dataset_name=dataset_name,
                batch_size=bsz, n_user=n_user, logger_fl=logger_fl, eval_output_path=eval_output_path
            )
            for i, uid in enumerate(it, start=1):
                torch.cuda.empty_cache()
                # a = InputExample(guid='0', instruction='Label text as Hateful or Non-hateful', text='I hate you!', prompt_examples="", label=["Hateful"], user_profile=None)
                # ts = ListDataset([a])#dset[uid].test)
                ts = ListDataset(dset[uid].test)

                path = os_join(model_name_or_path, uid2u_str(uid), 'trained')
                if len(ts) == 0 or (not zeroshot and not os.path.exists(path)):
                    logger.info(f'Skipping User {pl.i(uid)} due to missing trained model or empty test set...')
                    continue
                    
                if not zeroshot:  # load trained model for each user
                    # assert os.path.exists(path)  # sanity check
                    model, tokenizer = load_trained(model_name_or_path=path)

                accs[uid] = tester(model=model, dataset=ts, user_id=uid, user_idx=i)
                if not zeroshot:
                    model.cpu()  # move to CPU then collect memory, otherwise CUDA OOM error
                    gc.collect()
            out_args = dict(d_accs=accs, logger_fl=logger_fl, eval_output_path=eval_output_path, is_generative=is_generative)
            train_util.log_n_save_test_results(dataset_name=dataset_name, **out_args)

            t_e = tm.end()
            logger.info(f'Testing done in {pl.i(t_e)}')
            logger_fl.info(f'Testing done in {t_e}')
    command_prompt()

    def try_generate():
        md_nm = HF_MODEL_NAME
        # md_nm = 't5-base'
        # method = 'prefix'
        # method = 'p_tuning'
        method = 'prompt_tuning'
        model, tokenizer = load_model(model_name_or_path=md_nm, peft_method=method)
        model.eval()
        model.peft_config['default'].inference_mode = True
        mic(model.peft_config)
        mic(type(model))

        text = 'Is the following test happy or not. Say `yes` or `no`. I am happy.'
        inputs = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt')
        # mic(inputs)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)  # Greedy decoding
        lst_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        mic(lst_decoded)
    # try_generate()

    def check_learnable_param():
        # method = 'lora'
        # method = 'prefix'
        # method = 'p_tuning'
        method = 'prompt_tuning'
        model, tokenizer = load_model(HF_MODEL_NAME, peft_method=method)
        mic(get_trainable_param_meta(model, fmt='int'))
    # check_learnable_param()

    def chore_remove_user_folder_nms():
        dnm = 'cockamamie'
        # strt = 1731
        strt = 1705
        _, it = _get_dataset_and_users_it(dataset_name=dnm, uid_start_from=strt, use_user_profile=use_user_profile)
        # mic(it)
        x = ' '.join(uid2u_str(uid) for uid in it)
        print(x)
    # chore_remove_user_folder_nms()

