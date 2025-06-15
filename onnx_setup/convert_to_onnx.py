import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def convert_to_onnx(model_name, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create dummy inputs for the model
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        opset_version=17,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        export_params=True
    )
    return tokenizer

model_name = "microsoft/DialoGPT-large"
onnx_path = "models/microsoft_DialoGPT-large.onnx"
tokenizer_path = "models/microsoft_DialoGPT-large_tokenizer"

os.makedirs("models", exist_ok=True)
tokenizer = convert_to_onnx(model_name, onnx_path)
tokenizer.save_pretrained(tokenizer_path)
print(f"Model converted to ONNX and saved at {onnx_path}")