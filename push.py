from app import initialize_model,generate_output
from peft import (
        PeftModel
    )


def push_model_to_hub(model_name,lora_ckpt_dir,load_in_8bit):
    model,tokenizer = initialize_model(model_name,load_in_8bit)
    peft_model = PeftModel.from_pretrained(model,lora_ckpt_dir)
    peft_model = peft_model.merge_and_unload()

    generate_output(peft_model,"""
    Pitt: Hey Teddy! Have you received my message?                                                       
    Teddy: No. An email?                               
    Pitt: No. On the FB messenger.                     
    Teddy: Let me check.                               
    Teddy: Yeah. Ta!                                   
    Summary: 
        """,tokenizer)


    peft_model.push_to_hub("llama-7b-samsum")
    tokenizer.push_to_hub("llama-7b-samsum")


if __name__ == "__main__":
    push_model_to_hub("meta-llama/Llama-2-7b-hf","op/checkpoint-50",True)

