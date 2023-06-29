import os
from os.path import join as os_join
from typing import Tuple, List, Dict, Union
from logging import Logger
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer
)
from peft import get_peft_model, PeftConfig, PeftModel, PeftModelForSeq2SeqLM
from peft import TaskType, LoraConfig,  PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from peft import PromptTuningInit, PromptEncoderReparameterizationType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from stefutil import *
from peft_u.util import *
import peft_u.util.models as model_util
import peft_u.util.train as train_util
from peft_u.preprocess.load_dataset import *
from peft_u.trainer import HF_MODEL_NAME, get_arg_parser


logger = get_logger('PEFT Baseline')


PEFT_METHODS = ["lora", "prefix", "p_tuning", "prompt_tuning"]
DEFAULT_PEFT_METHOD = 'lora'


def parse_args():
    out = get_arg_parser(default_method=DEFAULT_PEFT_METHOD, method_choices=PEFT_METHODS)
    out.test_parser.add_argument("--zeroshot", type=bool, required=False, default=False)
    return out.parser.parse_args()


def load_model_n_tokenizer(
        model_name_or_path: str = HF_MODEL_NAME, peft_method: str = DEFAULT_PEFT_METHOD,
        verbose: bool = False, logger_fl: Logger = None
) -> Tuple[PeftModelForSeq2SeqLM, PreTrainedTokenizer]:
    cache_dir = model_util.get_hf_model_cache_dir()
    if verbose:
        logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')
    if logger_fl:
        logger_fl.info(f'Loading model {model_name_or_path} with cache dir {cache_dir}... ')

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = 512

    if peft_method is not None:
        ca.check_mismatch('PEFT method', peft_method, PEFT_METHODS)
        if verbose:
            logger.info('Loading Peft model...')
        if logger_fl:
            logger_fl.info('Loading Peft model...')

        config_d = dict(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False)
        if peft_method == 'lora':
            peft_config = LoraConfig(**config_d, r=8, lora_alpha=32, lora_dropout=0.1)
        else:
            _debug_larger_param = False
            _encoder_insertion_only = True
            # mic(_encoder_insertion_only)
            config_d['num_virtual_tokens'] = 32 if _debug_larger_param else 20
            if peft_method == 'prefix':
                peft_config = PrefixTuningConfig(**config_d)
            elif peft_method == 'p_tuning':
                peft_config: PeftConfig = PromptEncoderConfig(
                    **config_d, encoder_hidden_size=256 if _debug_larger_param else 128,
                    # originally MLP, the original paper uses LSTM
                    encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM
                    if _encoder_insertion_only else PromptEncoderReparameterizationType.MLP,
                    # for prompting in encoder input only
                    num_transformer_submodules=1 if _encoder_insertion_only else 2
                )
            else:
                assert peft_method == 'prompt_tuning'
                peft_config: PeftConfig = PromptTuningConfig(
                    **config_d, prompt_tuning_init=PromptTuningInit.RANDOM,
                    # for prompting in encoder input only
                    num_transformer_submodules=1 if _encoder_insertion_only else 2
                )
        model = get_peft_model(model, peft_config)

    model_meta = get_model_meta(model)
    if verbose:
        logger.info(f'Model info: {pl.i(model_meta)}')
    if logger_fl:
        logger_fl.info(f'Model info: {model_meta}')

    if torch.cuda.is_available():
        model.cuda()
    return model, tokenizer


def smart_batching_collate(batch, tokenizer):
    """
    Collate function for PyTorch DataLoader
    """
    return train_util.BatchCollator(tokenizer)(
        texts=[example.process_template() for example in batch],
        labels=[example.process_target() for example in batch]
    )


def _get_dataset_sizes(dataset: InputEgDataset):
    tr, vl, ts = dataset.train, dataset.val, dataset.test
    return dict(train_sz=len(tr), val_sz=len(vl), test_sz=len(ts))


def train_single(
        model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: InputEgDataset,
        seed: int = 42,
        batch_size: int = 8, num_epochs: int = 3, learning_rate: float = 2e-5, weight_decay: float = 0.01,
        output_path: str = None, user_id: str = None, verbose: bool = False, save_per_epoch: bool = True
):
    output_path = os_join(output_path, uid2u_str(user_id))

    def _save(dir_nm: str):
        model.save_pretrained(os_join(output_path, dir_nm))
        # tokenizer.save_pretrained(os_join(output_path, dir_nm))
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
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_item = loss.detach().item()
            total_tr_loss += loss_item
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            glob_step, lr = (epoch-1) * n_step_per_epoch + step, optimizer.param_groups[0]['lr']
            d_log = dict(epoch=epoch, step=glob_step, lr=lr, loss=loss_item)
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
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
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

        if save_per_epoch:
            _save(f'epoch_{epoch:02d}')
        if eval_epoch_loss < best_val_loss:
            best_val_loss = eval_epoch_loss
            _save('trained')

            best_val_loss_ = round(best_val_loss, 4)
            if verbose:
                logger.info(f'Best model saved w/ eval loss {pl.i(best_val_loss_)}')
            logger_fl.info(f'Best model saved w/ eval loss {best_val_loss_}')


def load_trained(model_name_or_path: str = None, verbose: bool = False) -> Tuple[PeftModel, PreTrainedTokenizer]:
    cache_dir = model_util.get_hf_model_cache_dir()
    if verbose:
        logger.info(f'Loading model {pl.i(model_name_or_path)} with cache dir {pl.i(cache_dir)}... ')

    config = PeftConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir)
    model = PeftModel.from_pretrained(model, model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


def test_single(
        model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: ListDataset = None, dataset_name: str = None,
        batch_size: int = 8, user_id: str = None, user_idx: int = None, n_user: int = None,
        logger_fl: Logger = None, eval_output_path: str = None
) -> float:
    label_options = sconfig(f'datasets.{dataset_name}.labels')
    n_sample = len(dataset)

    collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
    ts_dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    trues, preds = np.empty(n_sample, dtype=int), np.empty(n_sample, dtype=int)
    user_args = dict(user_id=user_id, user_idx=user_idx, n_user=n_user)
    it = train_util.get_user_test_pbar(it=ts_dl, **user_args)
    d_it = dict(dataset_size=pl.i(len(dataset)))
    it.set_postfix(d_it)

    ret = None
    n_ba = len(ts_dl)
    get_pred = train_util.GetPredId(label_options=label_options, logger_fl=logger_fl)
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
            out = get_pred(decoded=decoded, labels=dataset[i_sample].process_target(), user_id=user_id)
            preds[i_sample], trues[i_sample] = out.pred, out.true

        if i_ba + 1 == n_ba:  # last iteration
            ret = train_util.test_user_update_postfix_n_write_df(
                label_options=label_options, trues=trues, preds=preds, pbar=it, d_postfix=d_it,
                df_out_path=os_join(eval_output_path, f'{uid2u_str(user_id)}.csv')
            )
    return ret


def _get_dataset_and_users_it(
        dataset_name: str, leakage: bool = False, uid_start_from: Union[str, int] = None, seed: int = None
) -> Tuple[Dict[str, InputEgDataset], List[str]]:
    # from peft_u._dset_uid_too_small import uid_too_small

    dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage, seed=seed)

    filt = None
    # if dataset_name in uid_too_small:
    #     lst_filt = uid_too_small[dataset_name]
    #
    #     def filt(x):
    #         return x not in lst_filt
    it = iter_users(dataset=dset, start_from=uid_start_from, filter_fn=filt)
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
            output_path, logger_fl = train_util.setup_train_output_path_n_loggers(args=args, approach='peft')

            # strt = 23  # goemotion
            # strt = 28  # hatexplain
            # strt = 5021  # `measuringhatespeech.lora`
            # strt = 3896  # `measuringhatespeech.prefix`
            # strt = 3342  # `measuringhatespeech.p_tuning`
            # strt = 1161  # `measuringhatespeech.prompt_tuning`
            strt = 1678   # `cockamamie`
            # strt = 27  # `wikidetox`
            # strt = '45214884'  # `unhealthyconversations`
            # strt = None
            load_args = dict(dataset_name=dataset_name, leakage=leakage, seed=seed)
            dset, it = _get_dataset_and_users_it(**load_args, uid_start_from=strt)
            md_load_args = dict(peft_method=method, logger_fl=logger_fl)

            tm = Timer()
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
                    user_str_ordinal = train_util.get_user_str_w_ordinal(user_id=uid, user_idx=i, n_user=n_user)
                    logger.info(f'Launching {pl.i(dataset_name)} personalized training for User {user_str_ordinal}...')
                tm_ = Timer()

                # # if any dataset split is empty, skip
                # split_sizes = _get_dataset_sizes(dset[uid])
                # if any(v == 0 for v in split_sizes.values()):
                #     logger.info(f'Skipping User {pl.i(uid)} due to empty split w/ {pl.i(split_sizes)}...')
                #     continue

                model, tokenizer = load_model_n_tokenizer(model_name_or_path, **md_load_args)
                train_single(
                    model=model, tokenizer=tokenizer, dataset=dset[uid], seed=seed,
                    batch_size=args.batch_size, num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, output_path=output_path, user_id=uid, save_per_epoch=False
                )
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
            zeroshot = args.zeroshot

            date = now(fmt='short-date')
            if zeroshot:
                d = dict(md=model_util.hf_model_name_drop_org(model_name_or_path), dnm=dataset_name)
                eval_out_str = f'{date}_ZeroShot-Eval_{pl.pa(d)}'
            else:
                eval_out_str = f'{model_name_or_path}_Eval-{date}'
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
            model, tokenizer = None, None
            if zeroshot:  # Load global model for all users
                load_args = dict(verbose=True, logger_fl=logger_fl)
                model, tokenizer = load_model_n_tokenizer(model_name_or_path=model_name_or_path, **load_args)
                model.eval()
            else:
                model_name_or_path = model_util.prepend_local_model_path(model_path=model_name_or_path)

            # strt = 29044976  # unhealthyconversations
            strt = None
            load_args = dict(dataset_name=dataset_name, leakage=leakage, seed=seed)
            dset, it = _get_dataset_and_users_it(**load_args, uid_start_from=strt)
            n_user = len(it)
            d_log = dict(users=it, label_options=sconfig(f'datasets.{dataset_name}.labels'))
            logger.info(f'Testing w/ {pl.i(d_log)}...')
            logger_fl.info(f'Testing w/ {d_log}...')

            import gc
            accs = dict()
            for i, uid in enumerate(it, start=1):
                torch.cuda.empty_cache()
                ts = ListDataset(dset[uid].test)

                if not zeroshot:  # load trained model for each user
                    path = os_join(model_name_or_path, uid2u_str(uid), 'trained')
                    assert os.path.exists(path)  # sanity check
                    model, tokenizer = load_trained(model_name_or_path=path)
                # if len(ts) == 0:
                #     logger.info(f'Skipping User {pl.i(uid)} due to missing trained model or empty test set...')
                #     continue

                accs[uid] = test_single(
                    model=model, tokenizer=tokenizer, dataset=ts, batch_size=bsz, dataset_name=dataset_name,
                    user_id=uid, user_idx=i, n_user=n_user, logger_fl=logger_fl, eval_output_path=eval_output_path
                )
                model.cpu()  # move to CPU then collect memory, otherwise CUDA OOM error
                gc.collect()
            out_args = dict(d_accs=accs, logger_fl=logger_fl, eval_output_path=eval_output_path)
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
        model, tokenizer = load_model_n_tokenizer(model_name_or_path=md_nm, peft_method=method)
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
