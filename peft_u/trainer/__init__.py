from argparse import ArgumentParser

from peft_u.util import *


__all__ = ['HF_MODEL_NAME', 'parse_train_n_test_args']


HF_MODEL_NAME = 'google/flan-t5-base'

dataset_names = list(sconfig('datasets'))


def _add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model", type=str, required=False, default=HF_MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, required=True, choices=dataset_names)
    parser.add_argument("--leakage", type=str, required=False, default=True)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    return parser


def parse_train_n_test_args(train_parser: ArgumentParser, test_parser: ArgumentParser):
    train_parser = _add_args(train_parser)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    # Run on `cuda` if available, always personalize
    return train_parser, _add_args(test_parser)
