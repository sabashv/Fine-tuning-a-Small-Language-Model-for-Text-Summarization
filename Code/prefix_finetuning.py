from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import wandb
from utils import log_hardware, preprocess_function, compute_metrics, CustomCallback, preprocess_logits_for_metrics
import os
from peft import PrefixTuningConfig, get_peft_model
import time
import torch
import importlib
from torch.amp import GradScaler


def finetune_model(model_name, train_dataset, val_dataset, learning_rate=2e-5, batch_size=4, num_epochs=3, use_wandb=True, prompt="Summarize:",template=" ",one_shot_text=None, one_shot_summary=None,num_virtual_tokens=64,grad_acc_step=4,prefix_projection=False,OUTPUT_DIR="./"):


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prefix_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection= prefix_projection,
    )
    model = get_peft_model(model, prefix_config)

    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer, prompt,template, one_shot_text, one_shot_summary), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer, prompt, template, one_shot_text, one_shot_summary), batched=True)
    print("prefix fientunign")

    wandb_name = "prefix_"+ "lr:" + str(learning_rate) + "_vtokens:" + str(num_virtual_tokens) + "_batch:" + str(grad_acc_step*batch_size) + "_template:" + template
    if one_shot_text:
      wandb_name += "_1shot"
    else:
      wandb_name += "_0shot" 
    if prompt:
      wandb_name += "_prompt:" + prompt
    if prefix_projection:
      wandb_name += "_prefix_projection"

    OUTPUT_DIR = OUTPUT_DIR + "/finetune_prefix/" + wandb_name + "/" 
    

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
        warmup_steps=50,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        logging_steps=16,
        evaluation_strategy="steps",
        eval_steps=32,
        save_steps=256,
        save_total_limit=2,
        load_best_model_at_end=True,
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
    print("start finetuning")
    trainer.train()
    hardware_stats = log_hardware(start_time)
    custom_callback.plot_metrics(OUTPUT_DIR)
    if use_wandb and "WANDB_API_KEY" in os.environ:
        wandb.finish()

    with open(f"{OUTPUT_DIR}/hardware_stats.txt", "w") as f:
        f.write(str(hardware_stats))

    return trainer, hardware_stats
