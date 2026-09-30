from transformers import AutoModel, AutoTokenizer
import os

model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
output_dir = "models/onnx"

os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
print("Model files successfully prepared for ONNX deployment.")