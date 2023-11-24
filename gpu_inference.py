import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import sys
if len(sys.argv) != 3:
    print("Error: no file with prompt or model found")
    print("Usage: python3 gpu_inference.py prompt_file model_path")
    os._exit(1)


model_id = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load in 8bit
#model = AutoModelForCausalLM.from_pretrained(
#    model_id, 
#    load_in_8bit=True
#)

# load in 4bit
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_4bit=True
)

prompt =  open(sys.argv[2],"r").read()
encoded_input = tokenizer(prompt, return_tensors='pt')
#output = model(**encoded_input)
greedy_output = model.generate(**encoded_input, max_new_tokens=100)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True, stream=True))
