from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True,
    file_name="sentiment.onnx"
)

model.save_pretrained("models")
tokenizer.save_pretrained("models")

print("Export complete: models/sentiment.onnx")


