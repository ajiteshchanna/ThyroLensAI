"""
Diagnostic script: reveals the EXACT Python type and shape of every output
returned by grad_model() when called with the real FibonacciNet model.

Run BEFORE fixing gradcam.py so we know exactly what we're dealing with.
"""
import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.config import REPO_ID, MODEL_FILENAME

print("Loading real FibonacciNet model from HuggingFace cache...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
custom_objects = {
    "Avg2MaxPooling": Avg2MaxPooling,
    "DepthwiseSeparableConv": DepthwiseSeparableConv
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
print("Model loaded.")

# --- Find last DSC layer ---
last_conv = next(
    (l.name for l in model.layers[::-1] if "depthwise_separable_conv" in l.name.lower()),
    None
)
print(f"\nTarget Grad-CAM layer: {last_conv}")
target_layer = model.get_layer(last_conv)
print(f"Layer output type: {type(target_layer.output)}")
print(f"model.output type: {type(model.output)}")

# --- Build grad_model ---
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[target_layer.output, model.output]
)
print(f"\ngrad_model.output_names: {grad_model.output_names}")
print(f"Number of outputs: {len(grad_model.outputs)}")
for i, o in enumerate(grad_model.outputs):
    print(f"  output[{i}] shape: {o.shape}, type: {type(o)}")

# --- Call grad_model and inspect what comes back ---
dummy = np.zeros((1, 224, 224, 3), dtype='float32')

print("\n--- Calling grad_model(dummy) OUTSIDE GradientTape ---")
raw_out = grad_model(dummy)
print(f"Type of raw return: {type(raw_out)}")
print(f"Length: {len(raw_out)}")
for i, item in enumerate(raw_out):
    print(f"  raw_out[{i}]: type={type(item)}, shape/len={getattr(item, 'shape', len(item) if hasattr(item, '__len__') else 'N/A')}")

print("\n--- Calling grad_model(dummy) INSIDE GradientTape (tuple unpack) ---")
with tf.GradientTape() as tape:
    result = grad_model(dummy)

print(f"Type returned from tape context: {type(result)}")
print(f"Length: {len(result)}")
for i, item in enumerate(result):
    print(f"  result[{i}]: type={type(item)}, shape/len={getattr(item, 'shape', len(item) if hasattr(item, '__len__') else 'N/A')}")

# Unpack
conv_out = result[0]
preds = result[1]
print(f"\nconv_out type: {type(conv_out)}, shape: {getattr(conv_out, 'shape', 'N/A')}")
print(f"preds type:    {type(preds)}, shape: {getattr(preds, 'shape', 'N/A')}")

# Check if preds is list/tuple
if isinstance(preds, (list, tuple)):
    print(f"  --> preds is a {type(preds).__name__}! len={len(preds)}")
    print(f"  preds[0] type: {type(preds[0])}, shape: {getattr(preds[0], 'shape', 'N/A')}")
    preds_tensor = preds[0]
else:
    preds_tensor = preds
    print("  --> preds is already a tensor (good)")

print(f"\nFinal preds_tensor: {preds_tensor}")
print(f"preds_tensor[:, 0]: {preds_tensor[:, 0]}")
print("\nDiagnostic complete.")
