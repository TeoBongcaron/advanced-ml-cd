# FASTAPI app serving ONNX model
import onnxruntime as ort
import numpy as np

class SentimentModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, features: np.ndarray) -> float:
        # features shape: (1, N)
        outputs = self.session.run([self.output_name], {self.input_name: features})[0]
        # assume single scalar or single logit
        return float(outputs[0])

