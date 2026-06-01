# model_utils.py
import io
import os
from PIL import Image
import numpy as np
# import your actual ML dependencies here (tensorflow, torch, sklearn, etc.)
# Example: from your_training_code import load_model, predict_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "shade_model.pkl")

# -------------------------
# Replace this section with your own model loading & prediction
# -------------------------
_loaded_model = None

def load_model():
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    # if you have a pickle / keras model, load it here.
    # Example:
    # import joblib
    # _loaded_model = joblib.load(MODEL_PATH)
    # Or for TF: _loaded_model = tf.keras.models.load_model(...)
    # For now we keep a dummy predictor:
    _loaded_model = "DUMMY"
    return _loaded_model

def predict_from_image_pil(pil_image):
    """
    This function should take a PIL.Image and return the prediction result.
    Replace the body of this function with your actual preprocessing + model inference.
    Return format expected (example):
    {
      "sample_rgb": [210,170,150],
      "top_shades": [
         {"name":"Warm Sand","rgb":[210,170,150],"confidence":0.92},
         {"name":"Beige 02","rgb":[230,200,180],"confidence":0.84}
      ]
    }
    """
    # ---------- Demo/dummy implementation (replace with your real code) ----------
    # We'll take the center pixel median as a fake sample:
    img = pil_image.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    cy, cx = h//2, w//2
    sample_block = arr[max(0,cy-20):min(h,cy+20), max(0,cx-20):min(w,cx+20)]
    median_rgb = np.median(sample_block.reshape(-1,3), axis=0).astype(int).tolist()
    # dummy shades - replace with real shade lookup
    result = {
        "sample_rgb": median_rgb,
        "top_shades": [
            {"name":"Warm Sand", "rgb": median_rgb, "confidence": 0.93},
            {"name":"Beige 02", "rgb":[230,200,180], "confidence": 0.78},
            {"name":"Ivory 01", "rgb":[245,240,230], "confidence": 0.55},
        ]
    }
    return result

def predict_from_bytes(image_bytes):
    """Public helper: call this from Flask. Returns dict to jsonify."""
    model = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # optional: resize / preprocess according to how your model expects input
    # e.g. image = image.resize((224,224))
    pred = predict_from_image_pil(image)
    return pred
