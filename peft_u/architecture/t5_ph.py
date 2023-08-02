"""
Personalized head for T5, where an additional encoder layer is inserted after the encoders in the original T5
"""

import copy

from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, PreTrainedModel
from .t5_from_hf import T5Stack

from stefutil import *


__all__ = ['PHT5ForConditionalGeneration']


class PHT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # ========================== Begin of modified ==========================
        # Insert an additional encoder layer
        # Have to modify original init code cos need `post_init` call
        p_encoder_config = copy.deepcopy(config)
        p_encoder_config.num_layers = 1
        self.p_encoder = T5Stack(p_encoder_config, self.shared)
        # mic(encoder_config, p_encoder_config)
        # raise NotImplementedError
        # ========================== End of modified ==========================

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def freeze_all_but_ph(self):
        """
        modified from `peft.PeftModel::_setup_prompt_encoder`
        """
        transformer_backbone = None
        for name, module in self.named_children():
            # mic(name, isinstance(module, PreTrainedModel))
            if name != 'p_encoder':  # ignore the personalized head, see `__init__`
                for param in module.parameters():
                    param.requires_grad = False
                # TODO: not sure what this part does, the T5 encoder & decoder stacks passes
                if isinstance(module, PreTrainedModel):
                    # Make sure to freeze Transformers model
                    if transformer_backbone is None:
                        transformer_backbone = module
