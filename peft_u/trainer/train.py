import json
import math
from os.path import join as os_join
from typing import Tuple, Dict, Union, Callable
from logging import Logger
from argparse import Namespace
from functools import partial
from typing import List
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel
from transformers import Trainer, TrainingArguments, TrainerCallback, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stefutil import *
from peft_u.util import *
from peft_u.preprocess.load_dataset import *
import peft_u.util.models as model_util

from torch.cuda.amp import autocast #TODO fp16

if is_on_adapter():
    from transformers import AdapterTrainer

    PeftModel = None  # so that code runs for type hinting
else:
    import warnings
    from copy import deepcopy

    from peft import PromptLearningConfig, PeftModel, PeftModelForSeq2SeqLM
    from peft import PeftType, TaskType, MODEL_TYPE_TO_PEFT_MODEL_MAPPING


__all__ = [
    'HF_MODEL_NAME', 'get_arg_parser',
    'TqdmPostfixCallback',
    'setup_train_output_path_n_loggers', 'load_tokenizer', 'BatchCollator',
    'MyTrainer',
    'get_user_str_w_ordinal',
    'get_user_test_pbar',
    'PredIdPair', 'GetPredId',
    'test_user_update_postfix_n_write_df', 'MyTester', 'log_n_save_test_results'
]
if is_on_adapter():
    __all__ += ['MyAdapterTrainer']
else:
    __all__ += ['MyPeftModelForSeq2SeqLM']


logger = get_logger('Train Util')


HF_MODEL_NAME = 'google/flan-t5-base'

dataset_names = list(sconfig('datasets'))


def _add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model", type=str, required=False, default=HF_MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    parser.add_argument("--leakage", type=str, required=False, default=True)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    return parser


@dataclass
class ArgParser:
    parser: ArgumentParser = None
    train_parser: ArgumentParser = None
    test_parser: ArgumentParser = None


def get_arg_parser(default_method: str = None, method_choices: List[str] = None, has_method: bool = True) -> ArgParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser, test_parser = subparsers.add_parser('train'), subparsers.add_parser('test')
    if has_method:
        train_parser.add_argument(
            "--method", type=str, required=False, default=default_method, choices=method_choices)
    # otherwise case for personalized head
    train_parser = _add_args(train_parser)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    train_parser.add_argument("--use_user_profile", type=lambda x: (str(x).lower() == 'true'), required=False, default=False)
    # Run on `cuda` if available, always personalize
    return ArgParser(parser=parser, train_parser=train_parser, test_parser=_add_args(test_parser))


def load_tokenizer(model_name_or_path: str = HF_MODEL_NAME) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = 512
    return tokenizer


class BatchCollator:
    tok_args = dict(truncation=True, padding='max_length', return_tensors='pt')

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts: List[str], labels: List[str] = None):
        ret = self.tokenizer(texts, **BatchCollator.tok_args)
        labels = self.tokenizer(labels, **BatchCollator.tok_args)['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100  # `-100` is ignored in loss
        ret['labels'] = labels
        return ret


def smart_batching_collate(batch: List, tokenizer: PreTrainedTokenizer):
    """
    Collate function for PyTorch DataLoader
    """
    return BatchCollator(tokenizer)(
        texts=[example.process_template() for example in batch],
        labels=[example.process_target() for example in batch]
    )


class MyTrainer:
    def __init__(
            self, tokenizer: PreTrainedTokenizer, seed: int = 42,
            batch_size: int = 8, num_epochs: int = 3, learning_rate: float = 2e-5, weight_decay: float = 0.01,
            output_path: str = None, saver_cls=None,
    ):
        self.tokenizer = tokenizer
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.output_path = output_path
        self.saver_cls = saver_cls

    def __call__(
            self, model: torch.nn.Module, dataset: InputEgDataset,
            user_id: str = None, verbose: bool = False, save_per_epoch: bool = True
    ):
        output_path = os_join(self.output_path, uid2u_str(user_id))
        saver = self.saver_cls(model=model, output_base_path=output_path)

        set_seed(self.seed)
        collate_fn = partial(smart_batching_collate, tokenizer=self.tokenizer)
        tr, vl, ts = dataset.train, dataset.val, dataset.test

        train_dataloader = DataLoader(ListDataset(tr), batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(ListDataset(vl), batch_size=self.batch_size, collate_fn=collate_fn)

        # params = list(model.parameters())
        # mic(params, len(params))
        # raise NotImplementedError
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        n_step_per_epoch = len(train_dataloader)
        n_step = (n_step_per_epoch * self.num_epochs)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(n_step * 0.1),
            num_training_steps=n_step,
        )

        fp = os_join(output_path, 'train.log')
        logger_fl = get_logger(f'PEFT Train User-{user_id}', kind='file-write', file_path=fp)
        tb_writer = SummaryWriter(os_join(output_path, f'tensorboard'))
        d_log = get_dataset_sizes(dataset)
        logger.info(f'Dataset sizes: {pl.i(d_log)}')
        logger_fl.info(f'Dataset sizes: {d_log}')

        pret = MlPrettier(ref=dict(step=n_step_per_epoch, epoch=self.num_epochs), metric_keys=['cls_acc'])
        ls = LogStep(prettier=pret, logger=logger, file_logger=logger_fl, tb_writer=tb_writer)

        best_val_loss = float('inf')

        for epoch in range(1, self.num_epochs+1):
            model.train()
            total_tr_loss = 0

            tr_desc = f'Train Epoch {pl.i(epoch)}/{pl.i(self.num_epochs)}'
            it = tqdm(enumerate(train_dataloader, start=1), desc=tr_desc, total=n_step_per_epoch)
            for step, batch in it:
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                with autocast(): #TODO fp16
                    outputs = model(**batch)
                    loss = outputs.loss
                    # print('model==\n',model)
                    print('batch==',batch)
                    print('loss==',loss)
                    # exit(0)
                    loss_item = loss.detach().item()
                    print('loss_item==',loss_item)
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
            vl_desc = f'Eval Epoch {pl.i(epoch)}/{pl.i(self.num_epochs)}'

            n_vl_step = len(val_dataloader)
            it = tqdm(enumerate(val_dataloader), desc=vl_desc, total=n_vl_step)

            eval_epoch_loss = None
            for step, batch in it:
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                cum_eval_loss += outputs.loss.detach().item()

                decoded = self.tokenizer.batch_decode(
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
                saver(f'epoch_{epoch:02d}')
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                saver('trained')

                best_val_loss_ = round(best_val_loss, 4)
                if verbose:
                    logger.info(f'Best model saved w/ eval loss {pl.i(best_val_loss_)}')
                logger_fl.info(f'Best model saved w/ eval loss {best_val_loss_}')


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


if is_on_adapter():
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
else:
    class MyPeftModelForSeq2SeqLM(PeftModelForSeq2SeqLM):
        """
        Override `NotImplementedError` for `generate`
        """

        def generate(self, **kwargs):
            peft_config = self.active_peft_config
            self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self._prepare_encoder_decoder_kwargs_for_generation
            )
            try:
                if not isinstance(peft_config, PromptLearningConfig):
                    outputs = self.base_model.generate(**kwargs)
                else:
                    if "input_ids" not in kwargs:
                        raise ValueError("input_ids must be provided for Peft model generation")
                    if kwargs.get("position_ids", None) is not None:
                        warnings.warn(
                            "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                        )
                        kwargs["position_ids"] = None
                    if kwargs.get("token_type_ids", None) is not None:
                        warnings.warn(
                            "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                        )
                        kwargs["token_type_ids"] = None

                    if peft_config.peft_type == PeftType.PREFIX_TUNING:
                        outputs = self.base_model.generate(**kwargs)
                    else:
                        # ========================== Begin of modified ==========================
                        # raise NotImplementedError

                        use_peft_implementation = True
                        if use_peft_implementation:
                            assert peft_config.num_transformer_submodules == 1

                            assert peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]
                            kwargs = deepcopy(kwargs)
                            # Peft implementation
                            if "encoder_outputs" in kwargs:
                                del kwargs["encoder_outputs"]  # TODO: original peft code has typo
                                warnings.warn(
                                    "`encoder_outputs` should not be passed to `generate` when using prompt tuning. "
                                    "Ignoring it."
                                )

                            input_ids = kwargs.pop("input_ids")
                            inputs_embeds = self.word_embeddings(input_ids)
                            batch_size = inputs_embeds.shape[0]
                            prompts = self.get_prompt(batch_size=batch_size)
                            prompts = prompts.to(inputs_embeds.dtype)

                            inputs_embeds = torch.cat(
                                (prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                            kwargs["inputs_embeds"] = inputs_embeds

                            if "attention_mask" in kwargs:
                                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                                    kwargs["attention_mask"].device
                                )
                                kwargs["attention_mask"] = torch.cat(
                                    (prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                            outputs = self.base_model.generate(**kwargs)
                        else:  # My implementation
                            batch_size = kwargs["input_ids"].shape[0]
                            if kwargs.get("attention_mask", None) is not None:
                                # concat prompt attention mask
                                prefix_attention_mask = torch.ones(
                                    batch_size, peft_config.num_virtual_tokens).to(kwargs["input_ids"].device)
                                kwargs["attention_mask"] = torch.cat(
                                    (prefix_attention_mask, kwargs["attention_mask"]), dim=1).to(self.device)

                            prompts = self.get_prompt(batch_size=batch_size)
                            assert peft_config.num_transformer_submodules == 2
                            if kwargs.get("inputs_embeds", None) is None:
                                inputs_embeds = self.word_embeddings(kwargs["input_ids"].to(self.device))
                                prompts = prompts.to(inputs_embeds.dtype)
                                kwargs["inputs_embeds"] = torch.cat(
                                    (prompts[:, :peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                                kwargs["input_ids"] = None

                            if kwargs.get("decoder_inputs_embeds", None) is None \
                                    and kwargs.get("decoder_input_ids", None) is None:
                                decoder_input_ids_start = torch.ones(
                                    (batch_size, 1), dtype=torch.long, device=self.device)
                                decoder_input_ids_start *= self.config.decoder_start_token_id
                                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids_start)
                                decoder_inputs_embeds = torch.cat(
                                    (prompts[:, peft_config.num_virtual_tokens:], decoder_inputs_embeds), dim=1)
                                kwargs["decoder_inputs_embeds"] = decoder_inputs_embeds
                                kwargs["decoder_input_ids"] = None
                            outputs = self.base_model.generate(**kwargs)
                        # ========================== End of modified ==========================
            except:
                self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
                self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                    self.base_model_prepare_encoder_decoder_kwargs_for_generation
                )
                raise
            else:
                self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
                self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                    self.base_model_prepare_encoder_decoder_kwargs_for_generation
                )
                return outputs

    # so that `get_peft_model` uses the subclassed model
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING[TaskType.SEQ_2_SEQ_LM] = MyPeftModelForSeq2SeqLM


def setup_train_output_path_n_loggers(args: Namespace, approach: str = None) -> Tuple[str, Logger]:
    model_name_or_path, method = args.model, getattr(args, 'method', None)
    dataset_name, leakage = args.dataset_name, args.leakage
    batch_size, num_epochs, learning_rate = args.batch_size, args.num_epochs, args.learning_rate
    weight_decay = args.weight_decay
    seed = args.seed
    output_dir = args.output_dir
    use_user_profile = args.use_user_profile

    get_args = dict(model_name=model_name_or_path, name=output_dir, method=method, method_key=approach, use_user_profile=use_user_profile)
    output_path = model_util.get_train_output_path(**get_args, dataset_name=dataset_name)
    d_log = dict(
        model_name_or_path=model_name_or_path, method=method,
        dataset_name=dataset_name, leakage=leakage,
        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed,
        output_dir=output_dir, output_path=output_path, use_user_profile=use_user_profile
    )
    fnm = os_join(output_path, f'train_{now(for_path=True)}.log')
    cap_ap = approach.capitalize()
    logger_fl = get_logger(f'{cap_ap} Train fl', kind='file-write', file_path=fnm)
    logger.info(f'Training {pl.i(cap_ap)} w/ {pl.i(d_log)}...')
    logger_fl.info(f'Training {cap_ap} w/ {d_log}...')
    return output_path, logger_fl


def get_user_str_w_ordinal(user_id: str = None, user_idx: int = None, n_user: int = None):
    """
    Intended for terminal output
    """
    user_ordinal = f'{pl.i(user_idx)}/{pl.i(n_user)}'
    return f'{pl.i(user_id)}({user_ordinal})'


def get_user_test_pbar(it=None, user_id: str = None, user_idx: int = None, n_user: int = None, **kwargs):
    user_str = get_user_str_w_ordinal(user_id, user_idx, n_user)
    sep = pl.i('|')
    desc = f'{pl.i(now(for_path=True, color=True))}{sep}Testing on User {user_str}'
    return tqdm(it, desc=desc, **kwargs)


def _lenient_decoded(allowed_suffixes: List[str] = None, label_options: List[str] = None) -> Callable[[str], str]:
    """
    enforce an easier requirement by allowing no whitespace between labels,
        & being lenient here by dropping trailing full stop, quotes, verb suffixes, etc.

    If `label_options` is provided, and the decoded is a substring of one of the labels, that label is returned
    """
    def _ret(decoded: str) -> str:
        ret = decoded.strip()
        for suf in (allowed_suffixes or []):
            ret = ret.removesuffix(suf)

        for lb in (label_options or []):
            if ret in lb:
                return lb
        return ret
    return _ret


@dataclass
class PredIdPair:
    pred: int = None
    true: int = None


class GetPredId:
    def __init__(self, label_options: List[str], logger_fl: Logger = None):
        self.label_options = label_options
        self.lb2id = {lb: i for i, lb in enumerate(label_options)}  # sanity check each pred and true label is in config
        self.lenient = _lenient_decoded(allowed_suffixes=['.', "'", 'ing', 'ed', '"', ')'], label_options=label_options)

        self.logger_fl = logger_fl

    def __call__(self, decoded: str = None, labels: str = None, user_id: Union[str, int] = None) -> PredIdPair:
        labels = labels.split(', ')  # See `peft_u.preprocess.load_dataset::InputExample.process_target`
        # model may generate multiple labels
        decoded = [self.lenient(d) for d in decoded.split(',')]

        dec_not_in_lb = [dec for dec in decoded if dec not in self.label_options]
        if len(dec_not_in_lb) > 0:
            logger.warning(f'User {pl.i(user_id)} Predicted label(s) {pl.i(dec_not_in_lb)} not in label options')
            if self.logger_fl:
                self.logger_fl.warning(f'User {user_id} Predicted label(s) {dec_not_in_lb} not in label options')

            decoded = [dec for dec in decoded if dec in self.label_options]

            if len(decoded) == 0:  # doesn't generate anything in the label options, declare prediction wrong
                return PredIdPair(pred=-1, true=self.lb2id[labels[0]])

        # as of now, every predicted string is one of the label options
        for dec in decoded:
            if dec in labels:  # predicts one of the correct label, declare correct
                pred_id = self.lb2id[dec]
                return PredIdPair(pred=pred_id, true=pred_id)
        # if reached here, prediction wrong, arbitrarily default to first label
        return PredIdPair(pred=self.lb2id[decoded[0]], true=self.lb2id[labels[0]])


def test_user_update_postfix_n_write_df(
        label_options: List[str] = None, trues: np.ndarray = None, preds: np.ndarray = None,
        pbar: tqdm = None, d_postfix: Dict = None, df_out_path: str = None
) -> float:
    idx_lbs = list(range(len(label_options)))
    args = dict(
        labels=[-1, *idx_lbs], target_names=['Label not in dataset', *label_options],
        zero_division=0, output_dict=True
    )
    df, acc = eval_array2report_df(labels=trues, preds=preds, report_args=args, pretty=False)
    acc_str = f'{acc * 100:.1f}'
    d_postfix['cls_acc'] = pl.i(acc_str)
    pbar.set_postfix(d_postfix)

    df.to_csv(df_out_path)
    return acc


class MyTester:
    def __init__(
            self, tokenizer: PreTrainedTokenizer, dataset_name: str,
            batch_size: int = 8, n_user: int = None, logger_fl: Logger = None, eval_output_path: str = None
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        self.batch_size = batch_size
        self.n_user = n_user

        self.logger_fl = logger_fl
        self.eval_output_path = eval_output_path

    def __call__(
            self, model: Union[PreTrainedModel, PeftModel], dataset: ListDataset = None,
            user_id: str = None, user_idx: int = None,

    ) -> float:
        label_options = sconfig(f'datasets.{self.dataset_name}.labels')
        n_sample = len(dataset)

        collate_fn = partial(smart_batching_collate, tokenizer=self.tokenizer)
        ts_dl = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        trues, preds = np.empty(n_sample, dtype=int), np.empty(n_sample, dtype=int)
        user_args = dict(user_id=user_id, user_idx=user_idx, n_user=self.n_user)
        it = get_user_test_pbar(it=ts_dl, **user_args)
        d_it = dict(dataset_size=pl.i(len(dataset)))
        it.set_postfix(d_it)

        ret = None
        n_ba = len(ts_dl)
        get_pred = GetPredId(label_options=label_options, logger_fl=self.logger_fl)
        for i_ba, inputs in enumerate(it):
            inputs = {k: v for k, v in inputs.items()}
            inputs.pop('labels')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)  # Greedy decoding
            lst_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, decoded in enumerate(lst_decoded):
                i_sample = i_ba * self.batch_size + i
                out = get_pred(decoded=decoded, labels=dataset[i_sample].process_target(), user_id=user_id)
                preds[i_sample], trues[i_sample] = out.pred, out.true

            if i_ba + 1 == n_ba:  # last iteration
                ret = test_user_update_postfix_n_write_df(
                    label_options=label_options, trues=trues, preds=preds, pbar=it, d_postfix=d_it,
                    df_out_path=os_join(self.eval_output_path, f'{uid2u_str(user_id)}.csv')
                )
        return ret


def log_n_save_test_results(
        d_accs: Dict[str, float] = None, dataset_name: str = None, logger_fl: Logger = None,
        eval_output_path: str = None
):
    acc_avg = np.mean(list(d_accs.values()))
    acc_avg_str = f'{acc_avg * 100:.1f}'
    logger.info(f'Dataset {pl.i(dataset_name)} macro-avg acc: {pl.i(acc_avg_str)}')
    logger_fl.info(f'Dataset {dataset_name} macro-avg acc: {acc_avg_str}')
    with open(os_join(eval_output_path, 'accuracies.json'), 'w') as f:
        json.dump(d_accs, f, indent=4)


if __name__ == '__main__':
    def check_lenient_dec():
        label_options = [
            'answer', 'answer_overans-sway', 'cant-answer-lying', 'cant-answer-sincere',
            'shift-correct', 'shift-dodge'
        ]
        lenient = _lenient_decoded(allowed_suffixes=['.', "'", 'ing', 'ed'], label_options=label_options)
        dec = 'answer: cant-answer-sincere'
        mic(lenient(dec))
    check_lenient_dec()

