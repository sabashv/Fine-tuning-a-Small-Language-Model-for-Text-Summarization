# Fine-tuning-a-Small-Language-Model-for-Text-Summarization

This repository contains the code for fine-tuning a Small Language Model for Arabic text summarization. The project explores parameter-efficient fine-tuning techniques such as LoRA, QLoRA, and Prefix-Tuning, applied to a model (e.g., Qwen2.5-0.5B-Instruct) using a dataset of 5,000 Arabic news articles from the XLSum dataset. The fine-tuning was performed within the constraints of a Google Colab free-tier GPU (16GB VRAM), leveraging optimizations like quantization and mixed precision training.

The project evaluates model performance using ROUGE and BERTScore metrics, with findings indicating that LoRA offers an optimal balance between model quality and computational efficiency for resource-constrained environments.
