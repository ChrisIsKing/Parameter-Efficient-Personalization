from typing import List
from argparse import ArgumentParser
from dataclasses import dataclass

from peft_u.util import *


__all__ = ['HF_MODEL_NAME', 'get_arg_parser']


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


def get_arg_parser(default_method: str = None, method_choices: List[str] = None) -> ArgParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser, test_parser = subparsers.add_parser('train'), subparsers.add_parser('test')
    train_parser.add_argument(
        "--method", type=str, required=False, default=default_method, choices=method_choices)
    train_parser = _add_args(train_parser)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--output_dir", type=str, required=False, default=None)
    # Run on `cuda` if available, always personalize
    return ArgParser(parser=parser, train_parser=train_parser, test_parser=_add_args(test_parser))
