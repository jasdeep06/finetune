from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from app import generate_output,initialize_model
from peft import PeftModel

from fastapi import FastAPI

app = FastAPI()

model_id = "meta-llama/Llama-2-7b-hf"
lora_ckpt_dir = "op/checkpoint-50"
print("model id",model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id,load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
model,tokenizer = initialize_model(model_id,True)
peft_model = PeftModel.from_pretrained(model,lora_ckpt_dir)
peft_model = peft_model.merge_and_unload()

@app.post("/generate")
def read_generate(message: str):
    output = generate_output(peft_model,message,tokenizer)
    return output

