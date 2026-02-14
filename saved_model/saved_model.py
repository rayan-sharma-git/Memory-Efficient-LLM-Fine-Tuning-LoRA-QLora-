# using the saved lora adapter for prediction
# base model    -     TinyLlama-1.1B
# adapter       -     final_adapter from Tiny_Llama training

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# step-1 load base model and tokenizer
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path   = "./saved_model/Tiny_Llama/final_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# step-2 load lora adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# step-3 define prediction function
def predict(prompt, max_new_tokens=128, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# step-4 example usage
if __name__ == "__main__":
    test_prompt = (
        "### instruction:\nexplain the common symptoms of diabetes.\n\n### response:\n"
    )
    prediction = predict(test_prompt)
    print("\n===== model prediction =====\n")
    print(prediction)

# end of code