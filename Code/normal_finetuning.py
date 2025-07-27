from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import wandb
import optuna
from utils import log_hardware, preprocess_function, compute_metrics, CustomCallback, preprocess_logits_for_metrics
import os
import time
import torch
import importlib
from torch.amp import GradScaler



def finetune_model(model_name, train_dataset, val_dataset, learning_rate=2e-5, batch_size=4, num_epochs=3, use_wandb=True, prompt="Summarize:",template=" ",grad_acc_step=4,OUTPUT_DIR="./"):


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer, prompt,template, None, None), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer, prompt, template, None, None), batched=True)

    wandb_name = "normal_" + "lr:" + str(learning_rate) +  "_batch:" + str(grad_acc_step*batch_size) + "_template:" + template
  
    if prompt:
      wandb_name += "_prompt" + prompt
   

    print("normal finetuining:")

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
        warmup_steps=10,
        lr_scheduler_type="linear",
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

def run_optuna(model_name,train_dataset, val_dataset, hyperparams_to_tune, hyperparam_values, n_trials=5, use_wandb=True, prompt="Summarize:",OUTPUT_DIR="./"):
    def objective(trial):
        params = {}
        for param in hyperparams_to_tune:
            if param in hyperparam_values:
                params[param] = trial.suggest_categorical(param, hyperparam_values[param])
            else:
                raise ValueError(f"No values provided for {param}")

        params.setdefault("learning_rate", 2e-5)
        params.setdefault("batch_size", 2)
        params.setdefault("num_epochs", 3)

        trainer, _ = finetune_model(
            model_name,
            train_dataset,
            val_dataset,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            num_epochs=params["num_epochs"],
            use_wandb=use_wandb,
            prompt=prompt,
            OUTPUT_DIR=OUTPUT_DIR
        )
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best params: {study.best_params}")
    return study