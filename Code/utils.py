import torch
import time
import psutil
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from transformers import TrainerCallback
import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def log_hardware(start_time):
    elapsed_time = time.time() - start_time
    cpu_usage = psutil.cpu_percent()
    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    stats = {"time_s": elapsed_time, "cpu_percent": cpu_usage, "gpu_memory_gb": gpu_memory}
    print(f"Time: {elapsed_time:.2f}s, CPU: {cpu_usage:.2f}%, GPU Memory: {gpu_memory:.2f}GB")
    return stats

def preprocess_function(examples, tokenizer, prompt="Summarize:",template=" ", one_shot_text=None, one_shot_summary=None):
    if one_shot_summary:#few shot learning setting
   
      one_shot_example = template.format(text=one_shot_text, summary=one_shot_summary)
      if prompt:
        inputs = [f"{prompt}\n\n{one_shot_example}\n{template.format(text=text,summary='')}" for text in examples["text"]]
      else:
        inputs = [f"{one_shot_example}\n\n{template.format(text=text,summary='')}" for text in examples["text"]]
      targets = examples["summary"]
      full_texts = [f"{inp}{tgt}" for inp, tgt in zip(inputs, targets)]
      max_len = 640
    
    else:

      inputs = [f"{prompt}{template.format(text=text,summary='')}" for text in examples["text"]]
      targets = examples["summary"]
      full_texts = [f"{inp}{tgt}" for inp, tgt in zip(inputs, targets)]
      max_len = 510


   
    tokenized = tokenizer(full_texts, max_length=max_len, truncation=True, padding="max_length",return_tensors="pt")
    input_ids = tokenized["input_ids"]
   
    labels = input_ids.clone()  
   
    input_lengths = np.array([len(tokenizer(inp, add_special_tokens=False)["input_ids"]) for inp in inputs])
    
    mask = np.arange(labels.shape[1]) < input_lengths[:, None] 
    labels[mask] = -100
    
    model_inputs = {
        "input_ids": input_ids.tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
        "labels": labels.tolist()
    }
    return model_inputs

def preprocess_logits_for_metrics(logits, labels):

      if isinstance(logits, tuple):
          logits = logits[0]
      return logits.argmax(dim=-1)

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

class CustomCallback(TrainerCallback):
    def __init__(self, training_args=None):
        super().__init__()
        self.train_losses, self.val_losses = [], []
        self.val_accs = []
        self.steps = []
        self.training_args = training_args
    def on_train_begin(self, args, state, control, **kwargs):

        self.training_args = args
        if self.training_args:
            print("Hyperparameters:")
            hyperparams = {
                "learning_rate": self.training_args.learning_rate,
                "num_train_epochs": self.training_args.num_train_epochs,
                "per_device_train_batch_size": self.training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": self.training_args.per_device_eval_batch_size,
                "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps,
                "warmup_steps": self.training_args.warmup_steps,
                "weight_decay": self.training_args.weight_decay,
                "lr_scheduler_type": self.training_args.lr_scheduler_type,
                "eval_steps": self.training_args.eval_steps,
                "logging_steps": self.training_args.logging_steps
            }
            for key, value in hyperparams.items():
                print(f"{key}: {value}")
            print("-" * 50)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                print(f"Step {state.global_step}, Train Loss: {logs['loss']}")
            if "eval_loss" in logs:
                self.val_losses.append(logs["eval_loss"])
                print(f"Step {state.global_step}, Val Loss: {logs['eval_loss']}")
            if "eval_rouge1" in logs:
                self.val_accs.append(logs["eval_rouge1"])
                print(f"Step {state.global_step}, Val ROUGE-1: {logs['eval_rouge1']}")

    def plot_metrics(self, output_dir):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accs, label="Val ROUGE-1")
        plt.xlabel("Steps")
        plt.ylabel("ROUGE-1")
        plt.legend()
        plt.savefig(f"{output_dir}/metrics.png")
        plt.close()
