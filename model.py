import onnx
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType

# Загрузка модели
model_path = "./app/src/main/res/raw/mvit32_2.onnx"

quantized_model_path = "./app/src/main/res/raw/mvit32_2q.onnx"

# Выполнение квантизации

quantized_model = quantize_dynamic(
    model_path,
    quantized_model_path,
)

print(f"Квантизация завершена. Квантизированная модель сохранена в {quantized_model_path}")