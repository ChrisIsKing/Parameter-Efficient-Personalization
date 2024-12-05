from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import peft_u.trainer.train as train_util

from stefutil import *
from peft_u.util import *
import peft_u.util.models as model_util
import peft_u.trainer.train as train_util
from peft_u.preprocess.load_dataset import *
from peft_u.trainer import HF_MODEL_NAME, get_arg_parser

# Load model and tokenizer
model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model = AutoModelForCausalLM.from_pretrained(model_name)
print('1')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16) #TODO fp16)
print('2')
model.eval()
model.to('cuda')
print('3')

# Prepare input text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)
print('4')
# tester = train_util.MyTester(
#     tokenizer=tokenizer, dataset_name=dataset_name,
#     batch_size=bsz, n_user=n_user, logger_fl=logger_fl, eval_output_path=eval_output_path
# )

# torch.cuda.empty_cache()
# ts = ListDataset(dset[uid].test)

# path = os_join(model_name_or_path, uid2u_str(uid), 'trained')
# if len(ts) == 0 or (not zeroshot and not os.path.exists(path)):
#     logger.info(f'Skipping User {pl.i(uid)} due to missing trained model or empty test set...')
#     continue
    
# if not zeroshot:  # load trained model for each user
#     # assert os.path.exists(path)  # sanity check
#     model, tokenizer = load_trained(model_name_or_path=path)

# accs[uid] = tester(model=model, dataset=ts, user_id=uid, user_idx=i)

# Generate text with a decoding strategy
torch.cuda.empty_cache()
# ts = ListDataset(input_text)

# tester = train_util.MyTester(
#                 tokenizer=tokenizer, dataset_name='tweeteval'
#             )

if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

# inputs.pop('token_type_ids',None)

# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=128)  # Greedy decoding

# tester(model=model, dataset=ts, user_id="0", user_idx=0)

print(inputs)
with torch.no_grad():  # Disable gradient computation
    # outputs = model(inputs)
    outputs = model.generate(
        inputs["input_ids"],
        # max_length=50,
        # top_k=50,           # Use top-k sampling for variety
        # temperature=0.7,    # Set temperature for randomness
        # do_sample=True,
        max_new_tokens=10      # Enable sampling
    )
print('5')

# Decode and print the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)

# # Sanity check with a simple input
# simple_input_text = "The quick brown fox"
# simple_inputs = tokenizer(simple_input_text, return_tensors="pt")
# simple_outputs = model.generate(simple_inputs["input_ids"], max_length=20)
# simple_decoded_output = tokenizer.decode(simple_outputs[0], skip_special_tokens=True)
# print("Sanity Check Output:", simple_decoded_output)

# # Debugging: Print tokenized inputs and raw outputs
# print("Tokenized Inputs:", inputs["input_ids"])
# print("Raw Outputs:", outputs)
