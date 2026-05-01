# FASTAPI app serving ONNX model
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

class SentimentModel:
    def __init__(self, model_path: str):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Load tokenizer for DistilBERT SST-2
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

        # ONNX model expects these exact input names
        self.input_ids_name = self.session.get_inputs()[0].name
        self.attention_mask_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, text: str) -> float:
        # Convert text → numpy tensors with truncation to avoid ONNX shape errors
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512
        )

        ort_inputs = {
            self.input_ids_name: inputs["input_ids"],
            self.attention_mask_name: inputs["attention_mask"]
        }

        # Run ONNX inference
        outputs = self.session.run([self.output_name], ort_inputs)[0]

        # Flatten ANY output shape to 1D
        flat = outputs.flatten()

        # Use the last value (positive logit)
        score = float(flat[-1])

        return score




