# QLoRA Fine-Tuning: LLaMA-2 for Medical Question Answering

## About

Fine-tuning LLaMA-2-7B using QLoRA (4-bit quantization + LoRA adapters) on the MedQuad medical Q&A dataset.

## Objective

Perform domain-specific adaptation for medical question answering in a memory-efficient manner by training only LoRA adapters instead of full model weights.

## Model

Base model: meta-llama/Llama-2-7b-hf  
Method: QLoRA  
Quantization: 4-bit (NF4)  
Dataset: keivalya/MedQuad  

## Training

Run:

python train_qlora.py

The script:
- Loads the dataset from Hugging Face
- Formats question–answer pairs into instruction style
- Loads the model in 4-bit precision
- Applies LoRA adapters
- Trains adapter parameters only
- Saves trained adapters

## Output

Adapters are saved to:

saved_model/saved_model.py

Only LoRA adapter weights are stored.  
The base model is not duplicated.