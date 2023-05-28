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
    train_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--seed", type=int, required=False, default=42)
    train_parser.add_argument("--device", type=str, required=False, default="cuda")
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    train_parser.add_argument('--personalize', type=bool, required=False, default=True)

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
        batch_size: int = 8, tqdm_desc: str = None
) -> Tuple[pd.DataFrame, float]:
    label_options = sconfig(f'datasets.{dataset_name}.labels')
    lb2id = {lb: i for i, lb in enumerate(label_options)}  # sanity check each pred and true label is in config
    n_sample = len(dataset)

    collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
    ts_dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    trues, preds = np.empty(n_sample, dtype=int), np.empty(n_sample, dtype=int)
    it = tqdm(ts_dl, desc=tqdm_desc or 'Testing... ')
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

            tm = Timer()
            if personalize:
                # global_tqdm = True
                global_tqdm = False

                # strt = 47  # goemotion
                # strt = 108  # hatexplain
                strt = 127  # measuringhatespeech
                # strt = None

                uid_too_small = dict(  # TODO: for leakage==false only
                    goemotion=[47, 68],
                    hatexplain=[
                        40, 73, 74, 75, 86, 98, 102, 109, 117, 120, 134, 137, 138, 143, 144, 145, 147, 150, 151, 152,
                        154, 158, 159, 163, 164, 166, 168, 169, 170, 171, 174, 175, 177, 178, 179, 181, 182, 183, 184,
                        185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198
                    ],
                    measuringhatespeech=[
                        48, 64, 70, 71, 77, 82, 111, 118, 129, 132, 157, 174, 175, 176, 181, 184, 186, 187, 210, 216,
                        233, 234, 238, 239, 249, 276, 299, 301, 303, 305, 318, 327, 334, 347, 359, 361, 373, 382, 384,
                        397, 404, 405, 437, 440, 445, 454, 460, 469, 489, 495, 509, 514, 515, 520, 526, 545, 553, 555,
                        557, 559, 560, 577, 587, 590, 600, 609, 610, 629, 632, 665, 673, 676, 686, 689, 696, 703, 731,
                        736, 746, 766, 769, 798, 820, 829, 833, 835, 841, 846, 848, 872, 873, 884, 914, 916, 924, 931,
                        941, 950, 951, 965, 969, 978, 988, 992, 1018, 1039, 1044, 1048, 1058, 1060, 1080, 1082, 1090,
                        1093, 1118, 1121, 1123, 1164, 1175, 1181, 1199, 1214, 1217, 1218, 1229, 1233, 1245, 1258, 1259,
                        1266, 1280, 1314, 1337, 1352, 1367, 1368, 1371, 1387, 1392, 1393, 1397, 1399, 1403, 1408, 1427,
                        1451, 1458, 1459, 1466, 1469, 1472, 1479, 1485, 1487, 1488, 1512, 1515, 1527, 1533, 1538, 1542,
                        1549, 1564, 1575, 1586, 1589, 1593, 1596, 1603, 1607, 1617, 1618, 1629, 1635, 1637, 1642, 1655,
                        1667, 1673, 1680, 1689, 1691, 1705, 1718, 1744, 1745, 1746, 1757, 1771, 1778, 1782, 1788, 1789,
                        1796, 1810, 1817, 1821, 1827, 1843, 1855, 1885, 1886, 1892, 1900, 1933, 1943, 1949, 1962, 1968,
                        1971, 1972, 1979, 1986, 2019, 2024, 2032, 2040, 2052, 2073, 2074, 2076, 2079, 2094, 2102, 2104,
                        2108, 2109, 2111, 2126, 2137, 2161, 2165, 2167, 2171, 2174, 2184, 2191, 2219, 2232, 2240, 2266,
                        2282, 2289, 2293, 2297, 2300, 2315, 2323, 2325, 2344, 2355, 2362, 2363, 2369, 2385, 2389, 2395,
                        2401, 2410, 2411, 2412, 2421, 2422, 2431, 2440, 2446, 2451, 2455, 2480, 2484, 2496, 2498, 2499,
                        2505, 2507, 2509, 2513, 2529, 2542, 2564, 2619, 2624, 2649, 2661, 2663, 2672, 2675, 2679, 2687,
                        2714, 2719, 2736, 2746, 2766, 2768, 2775, 2790, 2794, 2814, 2820, 2836, 2844, 2848, 2851, 2854,
                        2861, 2867, 2872, 2875, 2877, 2889, 2890, 2894, 2895, 2897, 2948, 2951, 2971, 2984, 3004, 3008,
                        3044, 3054, 3058, 3069, 3079, 3091, 3097, 3098, 3100, 3125, 3128, 3130, 3135, 3138, 3171, 3178,
                        3186, 3199, 3204, 3207, 3215, 3231, 3239, 3246, 3250, 3252, 3275, 3292, 3293, 3302, 3306, 3312,
                        3315, 3332, 3336, 3337, 3356, 3372, 3374, 3375, 3376, 3403, 3406, 3434, 3441, 3445, 3456, 3459,
                        3464, 3465, 3471, 3480, 3487, 3499, 3502, 3512, 3516, 3522, 3530, 3532, 3535, 3542, 3568, 3582,
                        3585, 3587, 3591, 3599, 3638, 3648, 3649, 3650, 3655, 3696, 3708, 3718, 3732, 3739, 3741, 3750,
                        3757, 3766, 3769, 3779, 3782, 3793, 3796, 3797, 3804, 3815, 3817, 3821, 3828, 3868, 3873, 3896,
                        3921, 3929, 3933, 3934, 3939, 3943, 3945, 3950, 3954, 3959, 3961, 3971, 3973, 3998, 4005, 4012,
                        4030, 4031, 4058, 4063, 4065, 4066, 4071, 4094, 4095, 4100, 4109, 4124, 4140, 4151, 4154, 4168,
                        4176, 4201, 4215, 4232, 4237, 4238, 4258, 4265, 4268, 4292, 4318, 4341, 4346, 4357, 4364, 4376,
                        4381, 4392, 4413, 4414, 4440, 4448, 4458, 4476, 4477, 4517, 4531, 4535, 4594, 4597, 4599, 4607,
                        4617, 4628, 4636, 4643, 4660, 4676, 4677, 4683, 4688, 4699, 4705, 4717, 4727, 4734, 4737, 4743,
                        4747, 4748, 4754, 4795, 4799, 4800, 4811, 4828, 4837, 4846, 4874, 4881, 4887, 4889, 4914, 4918,
                        4922, 4926, 4932, 4935, 4939, 4945, 4946, 4975, 4991, 4993, 4998, 5014, 5030, 5033, 5034, 5039,
                        5043, 5075, 5082, 5083, 5084, 5113, 5122, 5132, 5158, 5159, 5163, 5171, 5182, 5192, 5193, 5222,
                        5247, 5253, 5267, 5277, 5287, 5293, 5303, 5319, 5320, 5325, 5331, 5342, 5360, 5368, 5379, 5380,
                        5389, 5393, 5404, 5408, 5422, 5423, 5424, 5445, 5466, 5487, 5493, 5500, 5507, 5517, 5520, 5531,
                        5536, 5543, 5544, 5545, 5546, 5572, 5579, 5582, 5594, 5603, 5612, 5613, 5620, 5622, 5655, 5666,
                        5673, 5675, 5676, 5684, 5690, 5691, 5697, 5702, 5704, 5720, 5754, 5765, 5786, 5794, 5798, 5812,
                        5824, 5852, 5857, 5866, 5867, 5871, 5872, 5883, 5886, 5890, 5901, 5902, 5928, 5932, 5935, 5944,
                        5945, 5958, 5972, 5983, 6001, 6003, 6007, 6026, 6027, 6031, 6036, 6038, 6070, 6108, 6113, 6123,
                        6139, 6157, 6165, 6173, 6175, 6176, 6184, 6185, 6186, 6228, 6242, 6252, 6256, 6266, 6278, 6303,
                        6317, 6334, 6341, 6358, 6363, 6377, 6402, 6413, 6428, 6433, 6452, 6464, 6465, 6479, 6484, 6492,
                        6493, 6515, 6520, 6533, 6537, 6546, 6572, 6584, 6585, 6603, 6629, 6635, 6638, 6643, 6647, 6651,
                        6666, 6668, 6669, 6674, 6694, 6695, 6699, 6700, 6716, 6721, 6723, 6746, 6749, 6789, 6792, 6814,
                        6826, 6846, 6856, 6866, 6880, 6884, 6931, 6949, 6956, 6957, 6980, 6982, 6986, 7006, 7012, 7013,
                        7020, 7026, 7029, 7031, 7082, 7084, 7092, 7104, 7106, 7107, 7130, 7140, 7150, 7153, 7158, 7162,
                        7172, 7175, 7180, 7186, 7194, 7198, 7202, 7203, 7208, 7210, 7213, 7228, 7243, 7268, 7286, 7289,
                        7307, 7309, 7321, 7325, 7326, 7328, 7344, 7352, 7353, 7369, 7372, 7376, 7377, 7384, 7385, 7388,
                        7392, 7422, 7446, 7454, 7460, 7462, 7478, 7482, 7485, 7486, 7496, 7505, 7506, 7519, 7531, 7555,
                        7558, 7568, 7575, 7580, 7592, 7593, 7610, 7620, 7631, 7652, 7654, 7665, 7681, 7708, 7735, 7763,
                        7769, 7777, 7783, 7811, 7825, 7843, 7853, 7854, 7858, 7864, 7868, 7870, 7892, 7893, 7895, 7905,
                        7909, 7912, 7923, 7941, 7949, 7960, 7968, 7971, 7985, 8000, 8001, 8003, 8009, 8021, 8027, 8030,
                        8048, 8050, 8054, 8055, 8057, 8059, 8075, 8093, 8122, 8123, 8138, 8146, 8159, 8166, 8179, 8184,
                        8185, 8198, 8209, 8222, 8227, 8229, 8247, 8252, 8260, 8307, 8334, 8340, 8342, 8343, 8348, 8352,
                        8367, 8368, 8380, 8388, 8422, 8439, 8441, 8447, 8458, 8466, 8473, 8479, 8483, 8494, 8495, 8507,
                        8517, 8519, 8524, 8531, 8543, 8544, 8553, 8556, 8560, 8562, 8564, 8566, 8582, 8587, 8589, 8594,
                        8606, 8635, 8648, 8652, 8678, 8687, 8696, 8702, 8710, 8717, 8719, 8750, 8756, 8763, 8779, 8785,
                        8787, 8798, 8823, 8827, 8828, 8862, 8866, 8882, 8895, 8909, 8913, 8922, 8936, 8965, 8975, 8987,
                        9000, 9002, 9014, 9017, 9032, 9035, 9043, 9060, 9066, 9067, 9081, 9084, 9088, 9100, 9101, 9102,
                        9109, 9113, 9124, 9153, 9172, 9179, 9190, 9200, 9213, 9216, 9219, 9220, 9226, 9232, 9236, 9251,
                        9264, 9267, 9272, 9280, 9285, 9295, 9302, 9316, 9328, 9338, 9355, 9364, 9395, 9408, 9411, 9426,
                        9429, 9435, 9443, 9452, 9463, 9470, 9477, 9480, 9486, 9497, 9499, 9526, 9538, 9540, 9548, 9549,
                        9556, 9559, 9566, 9572, 9573, 9585, 9595, 9596, 9604, 9607, 9624, 9629, 9641, 9645, 9656, 9662,
                        9663, 9668, 9681, 9689, 9704, 9709, 9714, 9715, 9719, 9731, 9748, 9763, 9776, 9793, 9811, 9829,
                        9833, 9836, 9844, 9857, 9864, 9866, 9900, 9906, 9911, 9914, 9928, 9931, 9942, 9952, 9967, 9972,
                        9980, 9996, 10000, 10006, 10007, 10010, 10020, 10027, 10028, 10043, 10044, 10064, 10075, 10090,
                        10096, 10103, 10105, 10124, 10131, 10133, 10141, 10170, 10171, 10183, 10218, 10229, 10235,
                        10237, 10247, 10251, 10269, 10270, 10280, 10292, 10297, 10302, 10308, 10310, 10314, 10340,
                        10367, 10375, 10387, 10389, 10396, 10401, 10402, 10425, 10431, 10434, 10437, 10439, 10454,
                        10459, 10472, 10474, 10475, 10504, 10509, 10516, 10536, 10541, 10544, 10549, 10565, 10571,
                        10575, 10584, 10593, 10595, 10601, 10606, 10626, 10643, 10650, 10668, 10691, 10706, 10710,
                        10723, 10730, 10733, 10769, 10774, 10795, 10804, 10812, 10820, 10873, 10877, 10895, 10899,
                        10915, 10933, 10943, 10954, 10965, 10978, 10988, 10997, 11003, 11010, 11021, 11029, 11031,
                        11042, 11045, 11049, 11064, 11077, 11082, 11084, 11089, 11091, 11094, 11108, 11117, 11118,
                        11120, 11124, 11131, 11135
                    ]
                )
                filt = None
                if dataset_name in uid_too_small:
                    lst_filt = uid_too_small[dataset_name]

                    def filt(x):
                        return int(x) not in lst_filt

                it = iter_users(dset, start_from=strt, filter_fn=filt)
                n_train = len(it)
                logger.info(f'Training on users {pl.i(it)}... ')
                logger_fl.info(f'Training on users {it}... ')
                if global_tqdm:
                    it = tqdm(it, desc=f'Training on {pl.i(dataset_name)}')

                for i, uid in enumerate(it, start=1):
                    # if uid != '1':  # TODO: debugging
                    #     continue
                    if global_tqdm:
                        d_log = dict(user=uid) | _get_dataset_sizes(dset[uid])
                        it.set_postfix({k: pl.i(v) for k, v in d_log.items()})
                    else:
                        user_ordinal = f'{pl.i(i)}/{pl.i(n_train)}'
                        logger.info(f'Launching {pl.i(dataset_name)} personalized training '
                                    f'for User {pl.i(uid)}({user_ordinal})...')
                    tm_ = Timer()

                    # reload model for each user
                    model, tokenizer = load_model_n_tokenizer(model_name_or_path, **load_args)
                    train_single(
                        model=model, tokenizer=tokenizer, dataset=dset[uid], device=device, seed=seed,
                        batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                        weight_decay=weight_decay, output_path=os_join(output_path, f'User-{uid}')
                    )
                    t_e_ = tm_.end()
                    if not global_tqdm:
                        logger.info(f'Training for User {pl.i(uid)} done in {pl.i(t_e_)}')
                    logger_fl.info(f'Training for User {uid} done in {t_e_}')
            else:
                model, tokenizer = load_model_n_tokenizer(model_name_or_path, **load_args, verbose=True)
                train_single(
                    model=model, tokenizer=tokenizer, dataset=dset, device=device,
                    batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                    weight_decay=weight_decay, output_path=output_path
                )
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

            if personalize:
                dset = load_dataset_with_prompts(dataset_name, leakage=leakage)
                model_name_or_path = os_join(get_base_path(), u.proj_dir, u.model_dir, model_name_or_path)

                accs = []

                for i, uid in enumerate(iter_users(dset), start=1):
                    ts = ListDataset(dset[uid].test)
                    # logger.info(f'Testing personalized PEFT for User {pl.i(uid)} w/ test split size {pl.i(len(ts))}...')
                    user_ordinal = f'{pl.i(i)}/{pl.i(len(dset))}'
                    desc = f'Testing on User {pl.i(uid)}({user_ordinal})... '

                    user_str = f'User-{uid}'
                    path = os_join(model_name_or_path, user_str, 'trained')
                    assert os.path.exists(path)  # sanity check
                    model, tokenizer = load_trained(model_name_or_path=path)

                    df, acc = test_single(
                        model=model, tokenizer=tokenizer, dataset=ts, batch_size=bsz,
                        dataset_name=dataset_name, tqdm_desc=desc
                    )
                    path = os_join(eval_output_path, f'{user_str}.csv')
                    df.to_csv(path)

                    accs.append(acc)
                acc_avg = np.mean(accs)
                logger.info(f'Dataset {pl.i(dataset_name)} macro-avg acc: {pl.i(acc_avg*100)}')
            else:
                raise NotImplementedError('Non personalized test')
    run()
