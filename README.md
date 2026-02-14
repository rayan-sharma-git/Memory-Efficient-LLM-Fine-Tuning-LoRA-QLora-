# Memory-Efficient Fine-Tuning of TinyLlama-1.1B using QLoRA

## Overview

This project fine-tunes a 1.1B parameter language model for medical question answering using QLoRA (4-bit quantization + LoRA adapters).

The base model is loaded in 4-bit precision and only LoRA adapter parameters are trained. This enables fine-tuning on a 4GB GPU without updating full model weights.

## Model

Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  
Method: QLoRA (4-bit NF4 quantization + LoRA)  
Dataset: medalpaca/medical_meadow_medqa  

## Approach

- Load base model in 4-bit precision using BitsAndBytes
- Freeze base weights
- Apply LoRA adapters
- Enable gradient checkpointing
- Train only adapter parameters
- Save trained adapters

This significantly reduces memory usage compared to full fine-tuning.

## Training

Run:

python train_qlora.py

The script:
- Loads and splits dataset
- Formats Q&A pairs
- Applies tokenization
- Injects LoRA adapters
- Trains adapter weights
- Saves final adapters

## Output

Trained LoRA adapters are saved in the folder `saved_model/Tiny_Llama/final_adapter`. To get predictions, use the Python file `saved_model/saved_model.py`.

The base model is not duplicated. Only adapter weights are stored.