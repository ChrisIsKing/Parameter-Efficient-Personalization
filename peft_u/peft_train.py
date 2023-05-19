import json
from argparse import ArgumentParser
from functools import partial

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, PrefixTuningConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_linear_schedule_with_warmup
)
from tqdm import tqdm

from data.utils import set_seed, process_data, instructions


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")

    train_parser.add_argument("--model", type=str, required=False, default="bigscience/mt0-xxl")
    train_parser.add_argument("--data", type=str, required=True)
    train_parser.add_argument("--task", type=str, required=True, choices=list(instructions.keys()))
    train_parser.add_argument("--method", type=str, required=False, default="lora", choices=["lora", "prefix"])
    train_parser.add_argument("--batch_size", type=int, required=False, default=8)
    train_parser.add_argument("--num_epochs", type=int, required=False, default=3)
    train_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    train_parser.add_argument("--seed", type=int, required=False, default=42)
    train_parser.add_argument("--device", type=str, required=False, default="cuda")
    train_parser.add_argument("--output_dir", type=str, required=True)

    test_parser.add_argument("--model", type=str, required=False, default="bigscience/mt0-xxl")
    test_parser.add_argument("--data", type=str, required=True)
    test_parser.add_argument("--batch_size", type=int, required=False, default=8)
    test_parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def smart_batching_collate(batch, tokenizer):
    """Collate function for PyTorch DataLoader."""
    inputs = [example.process_template() for example in batch]
    targets = [example.process_target() for example in batch]
    batch_encoding = tokenizer(inputs, truncation=False, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, truncation=True, padding="max_length", return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    batch_encoding["labels"] = labels
    return batch_encoding


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':

        batch_size = args.batch_size
        num_epochs = args.num_epochs
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        model_name_or_path = args.model
        data_path = args.data
        method = args.method
        output_dir = args.output_dir
        seed = args.seed
        task = args.task
        device = args.device
        set_seed(seed)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if method == 'lora':
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
        else:
            assert method == 'prefix'
            peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20
            )

        data = json.load(open(data_path, 'r'))

        train, val, test = process_data(data, task)

        warmup_steps = int(len(train) * num_epochs * 0.1)

        collate_fn = partial(smart_batching_collate, tokenizer=tokenizer)
        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(val_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(val_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            # Save the model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Evaluate on validation set
            correct = 0
            total = 0
            for pred, true in zip(eval_preds, val):
                if pred.strip() == true.process_target().strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            print(f"{accuracy=} % on the evaluation dataset")
