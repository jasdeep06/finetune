from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from app import generate_output,initialize_model
from peft import PeftModel
import time

from fastapi import FastAPI

app = FastAPI()

model_id = "meta-llama/Llama-2-7b-hf"
lora_ckpt_dir_8bit = "8-bit/checkpoint-500"

model_8bit,tokenizer_8bit = initialize_model(model_id,'8-bit')
peft_model_8bit = PeftModel.from_pretrained(model_8bit,lora_ckpt_dir_8bit)

peft_model_8bit = peft_model_8bit.merge_and_unload()

@app.post("/generate")
def read_generate(message: str):
    t1 = time.time()
    output,num_tokens = generate_output(peft_model_8bit,message,tokenizer_8bit,num_tokens=100)
    time_taken = round(time.time() - t1,2)
    return {'output':output,'time':time_taken,'num_tokens':num_tokens}



