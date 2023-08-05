import os
from os.path import join as os_join
from logging import Logger

import torch
import transformers

from stefutil import *
from peft_u.util import *
from peft_u.architecture import PHT5ForConditionalGeneration
from peft_u.preprocess.load_dataset import *
from peft_u.trainer import HF_MODEL_NAME, get_arg_parser
import peft_u.util.models as model_util
import peft_u.trainer.train as train_util


logger = get_logger('Personalized Head')

trained_ph_fnm = 'ph.bin'


def reduce_hf_logging():
    logger_ = transformers.logging.get_logger('transformers.modeling_utils')
    logger_.setLevel(transformers.logging.ERROR)  # Disables checkpoint not loading personalized head warning


def load_model(
        model_name_or_path: str = HF_MODEL_NAME, verbose: bool = False, logger_fl: Logger = None
) -> PHT5ForConditionalGeneration:
    log = model_util.LoadModelLogging(logger=logger, logger_fl=logger_fl, verbose=verbose)
    cache_dir = log.get_n_log_cache_dir(model_name_or_path=model_name_or_path)

    model = PHT5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model.freeze_all_but_ph()

    log.get_n_log_model_meta(model=model)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load_trained(model_name_or_path: str = HF_MODEL_NAME, verbose: bool = False) -> PHT5ForConditionalGeneration:
    model = load_model(model_name_or_path=HF_MODEL_NAME)
    if verbose:
        logger.info(f'Loading model from {pl.i(model_name_or_path)}')
    model.load_pretrained_ph(path=os_join(model_name_or_path, trained_ph_fnm))
    model.eval()
    return model


class TrainSaver:
    def __init__(self, model: PHT5ForConditionalGeneration, output_base_path: str = None, verbose: bool = False):
        self.model = model
        self.output_base_path = output_base_path
        self.verbose = verbose

    def __call__(self, output_dir_nm: str):
        out = os_join(self.output_base_path, output_dir_nm)
        os.makedirs(out, exist_ok=True)
        out = os_join(out, trained_ph_fnm)
        if self.model.insert_encoder_layer:
            torch.save(self.model.encoder.block[-1].state_dict(), out)
        else:  # insert encoder stack
            torch.save(self.model.encoder.state_dict(), out)
        if self.verbose:
            logger.info(f'Model Personalized Head saved to {pl.i(out)}')


if __name__ == '__main__':
    check_not_on_adapter()

    def try_load_model():
        model = PHT5ForConditionalGeneration.from_pretrained(HF_MODEL_NAME)
        # mic(model)
        mic(get_model_meta(model))
        model.freeze_all_but_ph()
        mic(get_model_meta(model))
    # try_load_model()

    def run():
        parser = get_arg_parser(has_method=False).parser
        args = parser.parse_args()
        cmd = args.mode
        if cmd == 'train':
            model_name_or_path = args.model
            dataset_name, leakage = args.dataset_name, args.leakage
            seed = args.seed
            output_path, logger_fl = train_util.setup_train_output_path_n_loggers(args=args, approach='PH')
            mic(output_path)

            dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage, seed=seed)
            strt = None
            it = iter_users(dataset=dset, start_from=strt)
            it = it[:4]  # TODO: debugging

            n_user = len(it)
            logger.info(f'Training on users {pl.i(it)}... ')
            logger_fl.info(f'Training on users {it}... ')

            tm = Timer()
            trainer = train_util.MyTrainer(
                tokenizer=train_util.load_tokenizer(),
                seed=seed, batch_size=args.batch_size, num_epochs=args.num_epochs,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                output_path=output_path, saver_cls=TrainSaver
            )
            for i, uid in enumerate(it, start=1):
                user_str_ordinal = train_util.get_user_str_w_ordinal(user_id=uid, user_idx=i, n_user=n_user)
                logger.info(f'Launching {pl.i(dataset_name)} personalized training for User {user_str_ordinal}...')

                tm_ = Timer()
                model = load_model(model_name_or_path=model_name_or_path, logger_fl=logger_fl)
                trainer(model=model, dataset=dset[uid], user_id=uid, save_per_epoch=False)
                t_e_ = tm_.end()
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

            date = now(fmt='short-date')
            eval_output_path = os_join(u.eval_path, f'{model_name_or_path}_Eval-{date}')
            os.makedirs(eval_output_path, exist_ok=True)
            d_log = dict(
                model_name_or_path=model_name_or_path, dataset_name=dataset_name, leakage=leakage,
                batch_size=bsz
            )
            logger.info(f'Testing PH w/ {pl.i(d_log)}...')
            fnm = os_join(eval_output_path, f'test_{now(for_path=True)}.log')
            logger_fl = get_logger('PH Test fl', kind='file-write', file_path=fnm)
            logger_fl.info(f'Testing PH w/ {d_log}...')

            tm = Timer()
            model_name_or_path = model_util.prepend_local_model_path(model_path=model_name_or_path)

            dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage, seed=seed)
            it = iter_users(dataset=dset)
            it = it[:4]  # TODO: debugging
            n_user = len(it)
            d_log = dict(users=it, label_options=sconfig(f'datasets.{dataset_name}.labels'))
            logger.info(f'Testing w/ {pl.i(d_log)}...')
            logger_fl.info(f'Testing w/ {d_log}...')

            accs = dict()
            tester = train_util.MyTester(
                tokenizer=train_util.load_tokenizer(), dataset_name=dataset_name,
                batch_size=bsz, n_user=n_user, logger_fl=logger_fl, eval_output_path=eval_output_path
            )
            for i, uid in enumerate(it, start=1):
                ts = ListDataset(dset[uid].test)
                path = os_join(model_name_or_path, uid2u_str(uid), 'trained')
                assert os.path.exists(path)  # sanity check
                model = load_trained(model_name_or_path=path)
                accs[uid] = tester(model=model, dataset=ts, user_id=uid, user_idx=i)
            out_args = dict(d_accs=accs, logger_fl=logger_fl, eval_output_path=eval_output_path)
            train_util.log_n_save_test_results(dataset_name=dataset_name, **out_args)

            t_e = tm.end()
            logger.info(f'Testing done in {pl.i(t_e)}')
            logger_fl.info(f'Testing done in {t_e}')
    run()
