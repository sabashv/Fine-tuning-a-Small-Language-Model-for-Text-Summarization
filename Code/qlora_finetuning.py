import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import wandb

from utils import log_hardware, preprocess_function, compute_metrics, CustomCallback,  preprocess_logits_for_metrics
import os
import time
def finetune_model(method,model_name,train_dataset, val_dataset, learning_rate=2e-5, batch_size=4, num_epochs=3, use_wandb=True, prompt="Summarize:", template=" ",one_shot_text=None, one_shot_summary=None, lora_r=32, grad_acc_step=4,OUTPUT_DIR="./"):
  

    if method == "qlora":
        print("qlora")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type='nf4',
          bnb_4bit_compute_dtype=compute_dtype,
          bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
          model_name,
          quantization_config=bnb_config,
          device_map="auto",
          trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
      
  
    elif method=="lora":
        print("lora")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    peft_config = LoraConfig(
      r=lora_r, lora_alpha=lora_r*2, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
      bias="none",
      task_type="CAUSAL_LM"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
   
    model = get_peft_model(model, peft_config)


    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer, prompt,template, one_shot_text, one_shot_summary), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer, prompt, template, one_shot_text, one_shot_summary), batched=True)

    wandb_name = method + "lr:" + str(learning_rate) + "_r:"+str(lora_r) + "_batch:" + str(grad_acc_step*batch_size) + "_template:" + template
  
    if prompt:
      wandb_name += "_prompt:" + prompt
   

    OUTPUT_DIR = OUTPUT_DIR + "/finetune_qlora/" + wandb_name + "/" 
    print("lora")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=grad_acc_step,
        fp16=True,
        eval_on_start=True,
        learning_rate=learning_rate,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        weight_decay=0.0001,
        logging_steps=16,
        evaluation_strategy="steps",
        eval_steps=32,
        save_steps=256,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="wandb" if use_wandb else "none",
        logging_dir=f"{OUTPUT_DIR}/logs",
    )

    custom_callback = CustomCallback()
  

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[custom_callback],
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )

    
    if use_wandb and "WANDB_API_KEY" in os.environ:
        wandb.init(project="NLP_finetuning", name=wandb_name)
    elif use_wandb:
        print("Wandb API key not found in environment. Skipping wandb logging.")
    start_time = time.time()
    trainer.train()
    hardware_stats = log_hardware(start_time)
    custom_callback.plot_metrics(OUTPUT_DIR)
    if use_wandb and "WANDB_API_KEY" in os.environ:
        wandb.finish()

    with open(f"{OUTPUT_DIR}/hardware_stats.txt", "w") as f:
        f.write(str(hardware_stats))

    return trainer, hardware_stats

