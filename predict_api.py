from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from app import generate_output,initialize_model

from fastapi import FastAPI

app = FastAPI()

model_id = "jasdeep06/llama-7b-samsum"
print("model id",model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id,load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
model,tokenizer = initialize_model(model_id,True)

@app.post("/generate")
def read_generate(message: str):
    output = generate_output(model,message,tokenizer)
    return output

