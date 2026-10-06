from transformers import AutoModel, AutoTokenizer
import os

model_id = "BAAI/bge-m3"
output_dir = "models/onnx"

os.makedirs(output_dir, exist_ok=True)

print(f"Downloading and preparing model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
print("Model files successfully prepared and saved for BGE-M3 deployment.")