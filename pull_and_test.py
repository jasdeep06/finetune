from app import generate_output,initialize_model,initialize_dataset,format_prompt


model_id = "jasdeep06/llama-7b-samsum"
print("model id",model_id)
model,tokenizer = initialize_model(model_id,True)
train_dataset,_,_ = initialize_dataset("samsum")
message = format_prompt(train_dataset[50]['dialogue'],'')
output = generate_output(model,message,tokenizer)
print(output)

