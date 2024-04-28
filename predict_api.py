from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from app import generate_output,initialize_model
from peft import PeftModel
import time
import math

from fastapi import FastAPI

app = FastAPI()

model_id = "meta-llama/Llama-2-7b-hf"
lora_ckpt_dir_4bit = "4-bit/checkpoint-500"
lora_ckpt_dir_8bit = "8-bit/checkpoint-500"

model_8bit,tokenizer_8bit = initialize_model(model_id,'8-bit')
model_4bit,tokenizer_4bit = initialize_model(model_id,'4-bit')
peft_model_8bit = PeftModel.from_pretrained(model_8bit,lora_ckpt_dir_8bit,load_in_8bit=True)
peft_model_4bit = PeftModel.from_pretrained(model_4bit,lora_ckpt_dir_4bit,load_in_4bit=True)

peft_model_8bit = peft_model_8bit.merge_and_unload()
peft_model_4bit = peft_model_4bit.merge_and_unload()

@app.post("/generate")
def read_generate(message: str,model:str):
    t1 = time.time()
    if model == '8-bit':
        output = generate_output(peft_model_8bit,message,tokenizer_8bit)
    elif model == '4-bit':
        output = generate_output(peft_model_8bit,message,tokenizer_4bit)
    time_taken = math.round(time.time() - t1,2)
    return output,time_taken



