from peft_u.architecture import PHT5ForConditionalGeneration
from peft_u.trainer import HF_MODEL_NAME

from stefutil import *
from peft_u.util import *


if __name__ == '__main__':
    check_not_on_adapter()

    model = PHT5ForConditionalGeneration.from_pretrained(HF_MODEL_NAME)
    # mic(model)
    mic(get_model_meta(model))
    model.freeze_all_but_ph()
    mic(get_model_meta(model))
