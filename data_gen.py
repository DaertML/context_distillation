import os
import requests
import json
import sys

if len(sys.argv) != 3 and  len(sys.argv) != 4:
    print("Error, provide the required parameters")
    print("Usage: python3 data_gen.py model prompts_file role")
    print("  - model: file path of the model that will generate data")
    print("  - prompt_file: file path of the prompts that will be used for the generated data")
    print("  - role: Providing a role is optional, by default all the roles will be done")
    os._exit(1)

def base_format_prompt(prompt, task=""):
    res = ""
    res += prompt["system"]
    if task != "":
       res += task
    else:
       res += prompt["task"]
    res += prompt["contents"]
    res += prompt["generation"]
    return res

def get_prompt_by_role(prompts, role):
    for pr in prompts:
        if pr["role"] == role:
            return pr

def llamacpp_generate(url, prompt, n_predict):
    header = {"Content-Type": "application/json"}
    data = {"prompt":prompt, "n_predict":n_predict}
    res = requests.post(url=url, data=json.dumps(data), headers=header)
    return json.loads(res.text)["content"]

model = sys.argv[1]
prompt_file = sys.argv[2]
role = ""
if len(sys.argv) == 4:
    role = sys.argv[3]

url = "http://localhost:8080/completion"
prompts = json.load(open(prompt_file, "r"))
n_gens = 1
n_predict = 1000

synth_data = {}
synth_data["model"] = model
synth_data["prompt_file"] = prompt_file
synth_data["generations"] = []

prompt_json = get_prompt_by_role(prompts, role)

if role == "":
    for prompt in prompts:
        role = prompt["role"]
        prompt_json = get_prompt_by_role(prompts, role)
        if prompt_json["task_type"] == "list":
            for task in prompt_json["task"]:
                prompt_txt = base_format_prompt(prompt_json, task)

                for i in range(n_gens):
                    print(prompt_txt)
                    generation = llamacpp_generate(url, prompt_txt, n_predict)
                    print(".....................")
                    print(generation)
                    print("---------------------")
                    gen_json = {}
                    gen_json["prompt"] = prompt_json
                    gen_json["output"] = generation
                    synth_data["generations"].append(gen_json)
        else:
            prompt_txt = base_format_prompt(prompt_json)

            for i in range(n_gens):
                print(prompt)
                generation = llamacpp_generate(url, prompt_txt, n_predict)
                print("....................")
                gen_json = {}
                print(generation)
                print("---------------------")
                gen_json["prompt"] = prompt_json
                gen_json["output"] = generation
                synth_data["generations"].append(gen_json)
else:
    prompt_json = get_prompt_by_role(prompts, role)
    if prompt_json["task_type"] == "list":
        for task in prompt_json["task"]:
            prompt_txt = base_format_prompt(prompt_json)

            for i in range(n_gens):
                print(prompt_txt)
                generation = llamacpp_generate(url, prompt_txt, n_predict)
                print("......................")
                print(generation)
                print("-------------------------")
                gen_json = {}
                gen_json["prompt"] = prompt_json
                gen_json["output"] = generation
                synth_data["generations"].append(gen_json)

json.dump(synth_data, open("output/distill.json", "w"))
