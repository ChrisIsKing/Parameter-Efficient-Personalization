from os.path import join as os_join
from typing import Tuple
from logging import Logger

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from stefutil import *
from peft_u.util import *
from peft_u.architecture import PHT5ForConditionalGeneration
from peft_u.preprocess.load_dataset import *
from peft_u.trainer import HF_MODEL_NAME, get_arg_parser
import peft_u.util.models as model_util
import peft_u.trainer.train as train_util


logger = get_logger('Personalized Head')


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


class TrainSaver:
    def __init__(self, model: PHT5ForConditionalGeneration, output_base_path: str = None, verbose: bool = False):
        self.model = model
        self.output_base_path = output_base_path
        self.verbose = verbose

    def __call__(self, output_dir_nm: str):
        out = os_join(self.output_base_path, output_dir_nm)
        torch.save(self.model.p_encoder.state_dict(), out)
        if self.verbose:
            logger.info(f'Model saved to {pl.i(out)}')


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
        model_name_or_path = args.model
        dataset_name, leakage = args.dataset_name, args.leakage
        seed = args.seed
        output_path, logger_fl = train_util.setup_train_output_path_n_loggers(args=args, approach='PH')
        mic(output_path)

        dset = load_dataset_with_prompts(dataset_name=dataset_name, leakage=leakage, seed=seed)
        strt = None
        it = iter_users(dataset=dset, start_from=strt)
        n_user = len(it)
        logger.info(f'Training on users {pl.i(it)}... ')
        logger_fl.info(f'Training on users {it}... ')

        tm = Timer()
        trainer = train_util.MyTrainer(
            tokenizer=train_util.load_tokenizer(),
            seed=seed, batch_size=args.batch_size, num_epochs=args.num_epochs,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            output_path=output_path
        )
        for i, uid in enumerate(it, start=1):
            user_str_ordinal = train_util.get_user_str_w_ordinal(user_id=uid, user_idx=i, n_user=n_user)
            logger.info(f'Launching {pl.i(dataset_name)} personalized training for User {user_str_ordinal}...')

            tm_ = Timer()
            model = load_model(model_name_or_path=model_name_or_path, verbose=True)
            saver = TrainSaver(model=model, output_base_path=output_path)
            trainer(model=model, dataset=dset[uid], user_id=uid, save_per_epoch=False, saver=saver)
            t_e_ = tm_.end()
            logger.info(f'Training for User {pl.i(uid)} done in {pl.i(t_e_)}')
            logger_fl.info(f'Training for User {uid} done in {t_e_}')
        t_e = tm.end()
        logger.info(f'Training done in {pl.i(t_e)}')
        logger_fl.info(f'Training done in {t_e}')
    run()
