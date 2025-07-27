from rouge_score import rouge_scorer
import bert_score
import numpy as np
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

def calc_bert_scroe(ground_truth_summaries, generated_summaries):
  P, R, F1 = bert_score.score(
    generated_summaries,
    ground_truth_summaries,
    lang="ar", 
    device="cuda",  
    verbose=True
  )

  avg_bert = {
    "Precision": P.mean().item(),
    "Recall": R.mean().item(),
    "F1": F1.mean().item()
  }
  print("\nAverage BERTScore:")
  for metric, score in avg_bert.items():
    print(f"{metric}: {score:.4f}")


def calc_rouge_score(ground_truth_summaries, generated_summaries):
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

  for gt, gen in zip(ground_truth_summaries, generated_summaries):
    scores = scorer.score(gt, gen)
    rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
    rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
    rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

  avg_rouge = {key: np.mean(scores) for key, scores in rouge_scores.items()}
  print("\nAverage ROUGE Scores:")
  for metric, score in avg_rouge.items():
    print(f"{metric}: {score:.4f}")
    

def run_evluation(test_dataset,model_name, checkpoint_path, template, output_split,batch_size, prompt=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to("cuda")
    summary_list = []
    for i in range(0, len(test_dataset), batch_size):
   
        batch = test_dataset[i:i + batch_size]  
        batch_inputs  = [f"{prompt}{template.format(text=text,summary='')}" for text in batch["text"]]
        inputs = tokenizer(
          batch_inputs,
          return_tensors="pt",
          padding=True,         
          truncation=True,     
          max_length=300       
        ).to("cuda")  


        model.eval()
        with torch.no_grad():
            outputs = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id  
        ).to("cuda")

        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for j, generated_summary in enumerate(batch_summaries):
          split_output = generated_summary.split(output_split)
          summ = split_output[1].strip() if len(split_output) > 1 else ""
          generated_summary_clean = tokenizer.decode(tokenizer.encode(summ), skip_special_tokens=True)
          summary_list.append(generated_summary_clean)
       
    print(f"\nTotal summaries generated: {len(summary_list)}")
    ground_truth_summaries = test_dataset['summary']
    calc_rouge_score(ground_truth_summaries, summary_list)
    calc_bert_scroe(ground_truth_summaries, summary_list)

