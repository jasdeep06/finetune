from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from app import generate_output

from fastapi import FastAPI

app = FastAPI()

model_id = "jasdeep06/llama-7b-samsum"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model)

@app.post("/generate")
def read_generate(message: str):
    output = generate_output(model,message,tokenizer)
    return output

    