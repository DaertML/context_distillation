import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

if len(sys.argv) != 3:
    print("Error: no file with prompt or model found")
    print("Usage: python3 cpu_inference.py prompt_file model_path")
    os._exit(1)

max_memory_mapping = {0: "10GB"}
model_path = sys.argv[1]

print("MODEL", model_path)
# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path, max_memory=max_memory_mapping,
    low_cpu_mem_usage=True,
    #device_map="auto"
    #load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-llama-2-7b")

prompt = open(sys.argv[2],"r").read()

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
outputs = model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Generated output:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
