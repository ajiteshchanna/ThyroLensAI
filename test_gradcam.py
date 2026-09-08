"""
Complete Grad-CAM pipeline test.
Tests both a synthetic tiny model AND the real FibonacciNet from HuggingFace.

Run with: .\\cenv\\Scripts\\python.exe test_gradcam.py
"""
import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
from PIL import Image

# ---------------------------------------------------------------
# TEST 1: matplotlib colormaps new API
# ---------------------------------------------------------------
import matplotlib
jet = matplotlib.colormaps["jet"]
dummy_hm = np.random.rand(7, 7).astype('float32')
hm_uint8 = np.uint8(255 * dummy_hm)
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[hm_uint8]
print(f"TEST 1 PASS: matplotlib.colormaps['jet'] | jet_heatmap shape: {jet_heatmap.shape}")

# ---------------------------------------------------------------
# TEST 2: save_and_display_gradcam — RGB input
# ---------------------------------------------------------------
from utils.gradcam import save_and_display_gradcam
rgb_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
result_rgb = save_and_display_gradcam(rgb_img, dummy_hm, alpha=0.4)
assert result_rgb.mode == "RGB"
print(f"TEST 2 PASS: RGB overlay | size={result_rgb.size}, mode={result_rgb.mode}")

# ---------------------------------------------------------------
# TEST 3: save_and_display_gradcam — grayscale input (like cancer.jpg)
# ---------------------------------------------------------------
gray_img = Image.fromarray(np.random.randint(0, 255, (352, 480), dtype=np.uint8), mode='L')
result_gray = save_and_display_gradcam(gray_img, dummy_hm, alpha=0.4)
assert result_gray.mode == "RGB"
print(f"TEST 3 PASS: Grayscale->RGB overlay | size={result_gray.size}, mode={result_gray.mode}")

# ---------------------------------------------------------------
# TEST 4: make_gradcam_heatmap — tiny SYNTHETIC model
# ---------------------------------------------------------------
import tensorflow as tf
from utils.gradcam import make_gradcam_heatmap

inp = tf.keras.layers.Input((8, 8, 3))
x = tf.keras.layers.Conv2D(16, 3, padding='same', name='depthwise_separable_conv_test')(inp)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
tiny_model = tf.keras.Model(inp, out)

dummy_input = np.random.rand(1, 8, 8, 3).astype('float32')
heatmap = make_gradcam_heatmap(dummy_input, tiny_model, 'depthwise_separable_conv_test')
assert heatmap is not None, "make_gradcam_heatmap returned None on tiny model!"
assert heatmap.shape == (8, 8), f"Expected (8,8), got {heatmap.shape}"
assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
print(f"TEST 4 PASS: Tiny model heatmap | shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")

# ---------------------------------------------------------------
# TEST 5: Full overlay -> base64 pipeline
# ---------------------------------------------------------------
import io, base64
overlay = save_and_display_gradcam(rgb_img, heatmap, alpha=0.4)
buf = io.BytesIO()
overlay.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
assert len(b64) > 100
print(f"TEST 5 PASS: Base64 encode | length={len(b64)} chars")

# ---------------------------------------------------------------
# TEST 6: Zero heatmap edge case
# ---------------------------------------------------------------
zero_heatmap = np.zeros((7, 7), dtype='float32')
result_zero = save_and_display_gradcam(rgb_img, zero_heatmap, alpha=0.4)
assert result_zero is not None
print(f"TEST 6 PASS: Zero heatmap handled | mode={result_zero.mode}")

# ---------------------------------------------------------------
# TEST 7: REAL FibonacciNet model — the critical test that catches
#         the preds-as-list bug that the synthetic test misses.
# ---------------------------------------------------------------
print("\nTEST 7: Loading REAL FibonacciNet from HuggingFace...")
from huggingface_hub import hf_hub_download
from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.config import REPO_ID, MODEL_FILENAME
from utils.processing import preprocess_image

model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
custom_objects = {
    "Avg2MaxPooling": Avg2MaxPooling,
    "DepthwiseSeparableConv": DepthwiseSeparableConv
}
real_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
print(f"  Real model loaded. Output type: {type(real_model.output)}")

# Find the actual last DSC layer
last_conv = next(
    (l.name for l in real_model.layers[::-1] if "depthwise_separable_conv" in l.name.lower()),
    None
)
assert last_conv is not None, "No depthwise_separable_conv layer found!"
print(f"  Target layer: {last_conv}")

# Use a grayscale-like test image (simulates cancer.jpg, mode=L -> convert to RGB in preprocess)
test_img = Image.fromarray(np.random.randint(50, 200, (352, 480), dtype=np.uint8), mode='L')
processed = preprocess_image(test_img)   # (1, 224, 224, 3) float32
assert processed.shape == (1, 224, 224, 3), f"Unexpected shape: {processed.shape}"

real_heatmap = make_gradcam_heatmap(processed, real_model, last_conv)
assert real_heatmap is not None, "make_gradcam_heatmap returned None on REAL model!"
assert real_heatmap.ndim == 2, f"Expected 2D heatmap, got shape {real_heatmap.shape}"
assert 0.0 <= real_heatmap.min() and real_heatmap.max() <= 1.0, \
    f"Heatmap out of [0,1]: min={real_heatmap.min()}, max={real_heatmap.max()}"
print(f"  Heatmap shape: {real_heatmap.shape}, range: [{real_heatmap.min():.3f}, {real_heatmap.max():.3f}]")

# Verify the full overlay also works with the real heatmap
real_overlay = save_and_display_gradcam(test_img, real_heatmap, alpha=0.4)
assert real_overlay.mode == "RGB"
buf2 = io.BytesIO()
real_overlay.save(buf2, format="PNG")
b64_real = base64.b64encode(buf2.getvalue()).decode("utf-8")
assert len(b64_real) > 100
print(f"  Real overlay base64 length: {len(b64_real)} chars")
print("TEST 7 PASS: REAL FibonacciNet Grad-CAM end-to-end OK")

print()
print("=" * 57)
print("ALL 7 TESTS PASSED — Grad-CAM pipeline is fully working.")
print("=" * 57)
