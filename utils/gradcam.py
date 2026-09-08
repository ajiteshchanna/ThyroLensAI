import tensorflow as tf
import numpy as np
import matplotlib
from PIL import Image
from utils.logger import logger


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.

    Gradient flow (standard Grad-CAM math):
      alpha_k = mean_spatial( d(score) / d(feature_map_k) )
      GradCAM = ReLU( sum_k( alpha_k * feature_map_k ) )  -> normalize to [0,1]

    Args:
        img_array            : preprocessed input batch (1, 224, 224, 3), float32
        model                : loaded FibonacciNet Keras model
        last_conv_layer_name : name of the final DSC layer before GlobalAvgPool
        pred_index           : class index to explain (None = 0 for binary sigmoid)

    Returns:
        numpy ndarray (H, W) float32 in [0, 1], or None on failure
    """
    # --- Build two-output sub-model: [conv_output, final_prediction] ---
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        logger.error(f"Grad-CAM: Failed to build sub-model for layer '{last_conv_layer_name}': {e}")
        return None

    # --- Forward pass inside GradientTape ---
    # IMPORTANT: In Keras 3, when model.output is a Python list (which happens with
    # certain saved models), grad_model() returns:
    #   result[0] = EagerTensor  (conv feature maps, shape 1x7x7x377)
    #   result[1] = Python list  [EagerTensor]  <-- NOT a bare tensor!
    # We must explicitly unwrap result[1] before doing [:, 0].
    with tf.GradientTape() as tape:
        model_out = grad_model(img_array)

        # Always index explicitly — do NOT use tuple unpacking on model_out
        # because result[1] may be a Python list, not a tensor.
        last_conv_layer_output = model_out[0]   # EagerTensor (1, 7, 7, 377)
        preds_raw = model_out[1]                # EagerTensor OR Python list[EagerTensor]

        tape.watch(last_conv_layer_output)

        # Normalise preds to a single EagerTensor regardless of Keras 3 wrapping.
        # Confirmed diagnosis: Keras 3 + this saved model returns preds_raw as list.
        if isinstance(preds_raw, (list, tuple)):
            preds_raw = preds_raw[0]            # unwrap the single-element list
        preds = tf.convert_to_tensor(preds_raw) # guarantee it is a tensor

        # Binary sigmoid: single output neuron, always index 0.
        # pred_index=0 → malignant probability score, shape (1,)
        class_channel = preds[:, 0]

    # --- Gradients: d(class_channel) / d(last_conv_layer_output) ---
    grads = tape.gradient(class_channel, last_conv_layer_output)

    if grads is None:
        logger.error(
            "Grad-CAM: GradientTape returned None. "
            "The target layer may not be in the computation path."
        )
        return None

    # --- alpha_k: global average pool over spatial dims (H, W) ---
    # grads: (1, H, W, C) -> pooled_grads: (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # --- Weighted sum of feature maps ---
    # last_conv_layer_output[0]: (H, W, C)
    # pooled_grads[..., tf.newaxis]: (C, 1)
    # @ result: (H, W, 1) -> squeeze -> (H, W)
    conv_out = last_conv_layer_output[0]                        # (H, W, C)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]          # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                               # (H, W)

    # --- ReLU ---
    heatmap = tf.maximum(heatmap, 0)

    # --- Safe normalization to [0, 1] ---
    max_val = tf.math.reduce_max(heatmap)
    if float(max_val) == 0.0:
        logger.warning("Grad-CAM: All heatmap activations are 0 after ReLU.")
        return heatmap.numpy()

    heatmap = heatmap / max_val
    return heatmap.numpy()  # (H, W) float32


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """
    Superimposes the Grad-CAM heatmap on the original image.

    Uses matplotlib 3.5+ colormaps API (colormaps['jet']) instead of the
    deprecated cm.get_cmap() which was removed in matplotlib 3.7+.

    Args:
        img     : PIL Image (any mode) or numpy uint8 array
        heatmap : numpy (H, W) float32, values in [0, 1]
        alpha   : heatmap blend weight (default 0.4)

    Returns:
        PIL.Image RGB of the superimposed Grad-CAM visualization
    """
    # --- Ensure RGB PIL Image ---
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img)  # (H, W, 3) uint8, original image

    # --- Apply 'jet' colormap ---
    # matplotlib.colormaps[name] is the modern API (matplotlib >= 3.5).
    # This replaces the removed cm.get_cmap(name) which broke on matplotlib 3.7+.
    jet = matplotlib.colormaps["jet"]

    heatmap_uint8 = np.uint8(255 * heatmap)        # (H_map, W_map) in [0, 255]
    jet_colors = jet(np.arange(256))[:, :3]         # (256, 3) RGB floats in [0,1]
    jet_heatmap = jet_colors[heatmap_uint8]          # (H_map, W_map, 3) floats [0,1]

    # --- Resize colored heatmap to original image size using PIL ---
    jet_heatmap_pil = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap_pil = jet_heatmap_pil.resize(
        (img_array.shape[1], img_array.shape[0]),   # PIL takes (width, height)
        resample=Image.BILINEAR
    )
    jet_heatmap_resized = np.array(jet_heatmap_pil).astype(np.float32)  # (H, W, 3) [0,255]

    # --- Blend heatmap over original ---
    img_float = img_array.astype(np.float32)
    superimposed = jet_heatmap_resized * alpha + img_float * (1.0 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)
