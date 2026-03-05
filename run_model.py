from transformers import AutoModel
from transformers import AutoTokenizer
from processor import Processor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("ngocson2002/vivqa-model", trust_remote_code=False).to(device)
processor = Processor()

image = Image.open('./ViVQA/demo/1.jpg').convert('RGB')
question = "màu áo của con chó là gì?"

inputs = processor(image, question, return_tensors='pt')
inputs["image"] = inputs["image"].unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(**inputs)
    logits = output.logits
    idx = logits.argmax(-1).item()

print("Predicted answer:", model.config.id2label[idx]) # prints: màu đỏ