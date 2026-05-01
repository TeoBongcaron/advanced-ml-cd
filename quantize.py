from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/sentiment.onnx",
    model_output="models/sentiment_quantized.onnx",
    weight_type=QuantType.QInt8
)

print("Quantized model saved as sentiment_quantized.onnx")
