from app import generate_output,initialize_model,initialize_dataset,format_prompt
import torch

model_id = "jasdeep06/llama-7b-samsum"
print("model id",model_id)
# model,tokenizer = initialize_model(model_id,True)
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jasdeep06/llama-7b-samsum")
model = AutoModelForCausalLM.from_pretrained("jasdeep06/llama-7b-samsum",torch_dtype=torch.float16)

train_dataset,_,_ = initialize_dataset("samsum")
message = format_prompt(train_dataset[50]['dialogue'],'')
output = generate_output(model,message,tokenizer)
print(output)

