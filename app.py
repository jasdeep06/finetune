import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig,
        PeftModel
    )
import transformers
from trl import SFTTrainer
import os


os.environ["WANDB_PROJECT"] = "llama-finetune"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def format_prompt(dialogue,summary,eos_token="</s>"):
    if bool(summary):
        summary += " " + eos_token
    return "Summerize the following:\n {} \n Summary: {}".format(dialogue,summary)   





# data = load_dataset("samsum")
# print(data["train"][0])

# print(format_prompt(data["train"][0]['dialogue'],data["train"][0]["summary"],"</s>"))

# prompt = format_prompt(data["train"][50]['dialogue'])

def generate_output(model,prompt,tokenizer):
    input_tokens = tokenizer(prompt,return_tensors="pt")["input_ids"].to("cuda")
    with torch.cuda.amp.autocast():
        output = model.generate(
            input_ids = input_tokens,
            max_new_tokens=1000,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.3,
            repetition_penalty=1.15,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    op = tokenizer.decode(output[0], skip_special_tokens=True)

    return op


def initialize_model(model_name,load_in_8bit):
    model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit=load_in_8bit,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

def initialize_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset['train'],dataset['test'],dataset['validation']

def formatting_func(example):
  output = []

  for d, s in zip(example["dialogue"], example["summary"]):
    op = format_prompt(d, s)
    output.append(op)

  return output

def load_peft_model(model,checkpoint_path):
    peft_model = PeftModel.from_pretrained(model,checkpoint_path,torch_dtype=torch.float16,offload_folder="results/temp")
    return peft_model

def finetune(model,tokenizer,r,train_data,val_data):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    output_dir = "op"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = 4
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 5e-4
    max_grad_norm = 0.3
    max_steps = 50
    warmup_ratio = 0.03
    evaluation_strategy="steps"
    lr_scheduler_type = "constant"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                evaluation_strategy=evaluation_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size,
                report_to='wandb'
            )
    
    trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args
    )
    
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()
    trainer.save_model(f"{output_dir}/final")


if __name__ == "__main__":
    train_dataset,test_dataset,val_dataset = initialize_dataset("samsum")
    model,tokenizer = initialize_model('meta-llama/Llama-2-7b-hf',load_in_8bit=False)
    sample_prompt = format_prompt(train_dataset[50]['dialogue'],'')
    print(sample_prompt)
    output = generate_output(model,sample_prompt,tokenizer)
    print("Output : ",output)
    finetune(model,tokenizer,8,test_dataset,val_dataset)
    peft_model = load_peft_model(model,'op/checkpoint-40')
    output = generate_output(peft_model,sample_prompt,tokenizer)
    print("Fine Tuned Output : ", output)


    





