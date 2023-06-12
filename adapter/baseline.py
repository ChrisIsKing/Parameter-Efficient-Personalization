"""
Try adapters from `adapter-transformers`

Note that since `adapter-transformers` is a direct fork on HF `transformers`
and we use a different `transformers` version, make sure to set up a separate environment for `peft_u` and `adapter`
"""

import math
import os
import json
from os.path import join as os_join

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HoulsbyConfig, T5TokenizerFast
from transformers.adapters import T5AdapterModel
from transformers import AdapterTrainer, TrainingArguments
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from stefutil import *


_DIRS = __file__.split(os.sep)
BASE_PATH, PROJ_DIR, PKG_DIR = os.sep.join(_DIRS[:-3]), _DIRS[-3], _DIRS[-2]
DSET_DIR = 'data'


if __name__ == '__main__':
    INSTR = "Please review the following text and indicate if it has the presence of hate speech. "\
            "Respond 'Hateful' if the text contains hate speech "\
            "and 'Non-hateful' if the text does not contain hate speech."

    def load_dset(tokenizer: T5TokenizerFast, tokenize: bool = True):
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
                # return {k: v.tolist() for k, v in ret.items()}
                return ret
            return dset.map(
                map_single, batched=True,
                remove_columns=['text', 'label']
            )
        else:
            return dset

    ADAPTER_NM = 'debug'
    DEBUG = True
    # MD_NM = 'google/flan-t5-base'
    MD_NM = 'google/flan-t5-small'

    def train():
        model = T5AdapterModel.from_pretrained(MD_NM)  # Should observe a warning on `lm_head.weight` not used

        model.add_adapter(adapter_name=ADAPTER_NM, config=HoulsbyConfig())
        model.add_seq2seq_lm_head(head_name=ADAPTER_NM)
        model.train_adapter(ADAPTER_NM)  # activate for training

        # Since there's not a LM head version of `T5AdapterModel`,
        # use the LM head from the HF model and set it to frozen, i.e. override `train_adapter` on the LM head
        model_dummy = AutoModelForSeq2SeqLM.from_pretrained(MD_NM)
        # mic(model_dummy)
        state_d = model_dummy.lm_head.state_dict()
        assert set(state_d.keys()) == {'weight'}  # sanity check
        pretrained_lm_weight = state_d['weight']
        lm_head = model.heads[model.active_head]
        assert len(lm_head._modules) == 1  # sanity check, should only contain `output_embeddings`
        assert pretrained_lm_weight.shape == lm_head.get_output_embeddings().weight.shape
        mic(lm_head.get_output_embeddings().weight)
        mic(lm_head.get_output_embeddings().weight.dtype)
        mic(type(lm_head.get_output_embeddings().weight))
        # raise NotImplementedError
        # lm_head.set_output_embeddings(torch.nn.Embedding.from_pretrained(pretrained_lm_weight, freeze=True))
        # self._modules[next(reversed(self._modules))][:] =
        lm_head_weight = lm_head.get_output_embeddings().weight
        lm_head_weight.requires_grad = False
        lm_head_weight[:] = pretrained_lm_weight
        del model_dummy, state_d

        mic(get_model_meta(model))
        tokenizer = AutoTokenizer.from_pretrained(MD_NM)
        tokenizer.model_max_length = 512

        dset = load_dset(tokenizer=tokenizer)
        mic(dset)
        # raise NotImplementedError

        date = now(fmt='short-date')
        _md_nm = MD_NM
        if '/' in _md_nm:
            org, _md_nm = _md_nm.split('/')
        meta = dict(md_nm=_md_nm, adapter='Houlsby')
        output_path = os_join(BASE_PATH, PROJ_DIR, 'models', f'{date}_{pl.pa(meta)}_{ADAPTER_NM}')
        os.makedirs(output_path, exist_ok=True)
        train_args = TrainingArguments(
            output_dir=output_path,
            do_train=True,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            num_train_epochs=2 if DEBUG else 8,
            remove_unused_columns=False
        )
        tr, vl = dset['train'], dset['validation']
        if DEBUG:
            tr = tr.select(range(16))
            vl = tr

        trainer = AdapterTrainer(model=model, args=train_args, tokenizer=tokenizer, train_dataset=tr, eval_dataset=vl)
        trainer.train()
        model.save_adapter(save_directory=output_path, adapter_name=ADAPTER_NM)
    train()

    def test():
        model = T5AdapterModel.from_pretrained(MD_NM)
        tokenizer = AutoTokenizer.from_pretrained(MD_NM)
        tokenizer.model_max_length = 512

        adapter_path = os_join(BASE_PATH, 'models', '23-06-08_{md_nm=flan-t5-small, adapter=Houlsby}_debug')
        model.load_adapter(adapter_name_or_path=adapter_path)
        model.load_head(save_directory=adapter_path)
        model.set_active_adapters(ADAPTER_NM)

        dset = load_dset(tokenizer=tokenizer)['test']
        idxs_gen = group_n(range(len(dset)), n=8)
        total = math.ceil(len(dset) / 8)
        for i_ba, idxs in enumerate(tqdm(idxs_gen, desc='Testing', total=total)):
            idxs = [int(idx) for idx in idxs]
            inputs = {k: torch.tensor(v) for k, v in dset[idxs].items() if k not in ['text', 'label', 'labels']}
            # mic(inputs)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=16)
            lst_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            mic(lst_decoded)
            raise NotImplementedError
    # test()
