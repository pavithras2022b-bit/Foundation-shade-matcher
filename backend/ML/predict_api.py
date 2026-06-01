# ML/predict_api.py
# Usage: python predict_api.py <image_path> <model_path>
# Example: python predict_api.py ../uploads/1699999999-myfile.jpg foundation_model.pkl

import sys
import json
import os
import joblib
import numpy as np
import cv2
from sklearn.metrics import pairwise_distances_argmin_min

def load_model_data(model_path):
    model_obj = joblib.load(model_path)
    # model_obj could be a dict or a tuple/list
    if isinstance(model_obj, dict):
        scaler = model_obj.get("scaler")
        model = model_obj.get("model")
        data = model_obj.get("data")
        # some versions used different keys
        if scaler is None or model is None or data is None:
            # try alternative keys
            scaler = model_obj.get("scaler") or model_obj.get("sc")
            model = model_obj.get("model") or model_obj.get("clf")
            data = model_obj.get("data") or model_obj.get("df")
    elif isinstance(model_obj, (list, tuple)):
        # common packing: (scaler, model, df) or (scaler,model)
        if len(model_obj) >= 3:
            scaler, model, data = model_obj[0], model_obj[1], model_obj[2]
        elif len(model_obj) == 2:
            scaler, model = model_obj
            data = None
        else:
            raise ValueError("Unexpected model object shape.")
    else:
        raise ValueError("Unsupported model object type.")
    return scaler, model, data

def extract_skin_rgb_simple(image_path):
    """Simple skin extraction: HSV threshold mask; returns average RGB (0-255)"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # HSV skin thresholds (tunable)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)
    skin_pixels = img_rgb[mask > 0]

    # fallback: if no pixels, take center crop
    if len(skin_pixels) == 0:
        h, w = img_rgb.shape[:2]
        ch, cw = h // 2, w // 2
        crop = img_rgb[max(0, ch-50):min(h, ch+50), max(0, cw-50):min(w, cw+50)]
        if crop.size == 0:
            # last fallback: whole image mean
            avg = img_rgb.reshape(-1, 3).mean(axis=0)
            return avg
        return crop.reshape(-1,3).mean(axis=0)

    return skin_pixels.mean(axis=0)

def recommend_from_model(avg_rgb, scaler, model, data):
    # robust handling if data missing
    dataset_rgbs = None
    shade_name = None
    if scaler is not None and model is not None:
        try:
            avg_scaled = scaler.transform([avg_rgb])
            ml_pred = model.predict(avg_scaled)[0]
        except Exception:
            ml_pred = None
    else:
        ml_pred = None

    if data is not None:
        # try to find dataset RGBs (col names tolerant)
        df = data
        cols = [c for c in df.columns if c.lower() in ("r","g","b","R","G","B","r "," g "," b ")]
        # safer: try many names
        if set(["R","G","B"]).issubset(df.columns):
            dataset_rgbs = df[["R","G","B"]].values.astype(float)
        else:
            # try lowercase
            cands = []
            for name in ["r","g","b","R","G","B","r "," g "," b "]:
                if name in df.columns:
                    cands.append(name)
            if len(cands) >= 3:
                dataset_rgbs = df[cands[:3]].values.astype(float)
        # try to get rule-based (closest in dataset)
        if dataset_rgbs is not None:
            idx, dists = pairwise_distances_argmin_min([avg_rgb], dataset_rgbs)
            rule_idx = idx[0]
            rule_pred = None
            # find shade_name column possible names
            shade_cols = [c for c in df.columns if c.lower() in ("shade_name","shade","name")]
            if shade_cols:
                rule_pred = df.iloc[rule_idx][shade_cols[0]]
            else:
                # fallback to first text column
                text_cols = df.select_dtypes(include=["object"]).columns
                if len(text_cols)>0:
                    rule_pred = df.iloc[rule_idx][text_cols[0]]
            # build final selection
        else:
            rule_pred = None
    else:
        rule_pred = None

    # Hybrid decision
    final_shade = None
    if ml_pred is not None and rule_pred is not None:
        # compute distances
        try:
            if dataset_rgbs is not None and ml_pred is not None:
                # compute mean RGB of ml_pred shade rows if present
                rows = data[data.apply(lambda r: r.astype(str).str.contains(str(ml_pred), case=False).any(), axis=1)]
                if not rows.empty and set(["R","G","B"]).issubset(data.columns):
                    mean_ml_rgb = rows[["R","G","B"]].astype(float).mean().values
                    dist_ml = np.linalg.norm(avg_rgb - mean_ml_rgb)
                else:
                    dist_ml = 1e9
                dist_rule = np.linalg.norm(avg_rgb - dataset_rgbs[rule_idx])
                final_shade = ml_pred if dist_ml < dist_rule else rule_pred
            else:
                final_shade = ml_pred or rule_pred
        except Exception:
            final_shade = ml_pred or rule_pred
    else:
        final_shade = ml_pred or rule_pred or "Unknown"

    # now get shade info row
    shade_info = {}
    if data is not None:
        # try to find the row(s) that match final_shade
        shade_cols = [c for c in data.columns if c.lower() in ("shade_name","shade","name")]
        if shade_cols:
            rows = data[data[shade_cols[0]].astype(str).str.strip().str.lower() == str(final_shade).strip().lower()]
            if rows.empty:
                # try contains
                rows = data[data[shade_cols[0]].astype(str).str.lower().str.contains(str(final_shade).strip().lower())]
            if not rows.empty:
                row = rows.iloc[0].to_dict()
                shade_info = row

    return final_shade, shade_info

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error":"usage: python predict_api.py <image_path> <model_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(json.dumps({"error":f"image not found: {image_path}"}))
        sys.exit(1)
    if not os.path.exists(model_path):
        print(json.dumps({"error":f"model not found: {model_path}"}))
        sys.exit(1)

    try:
        scaler, model, data = load_model_data(model_path)
    except Exception as e:
        print(json.dumps({"error":f"loading model failed: {str(e)}"}))
        sys.exit(1)

    try:
        avg_rgb = extract_skin_rgb_simple(image_path)  # numpy array [R,G,B]
    except Exception as e:
        print(json.dumps({"error":f"skin extraction failed: {str(e)}"}))
        sys.exit(1)

    try:
        final_shade, shade_info = recommend_from_model(avg_rgb, scaler, model, data)
    except Exception as e:
        print(json.dumps({"error":f"recommendation failed: {str(e)}"}))
        sys.exit(1)

    out = {
        "predicted_shade": str(final_shade),
        "avg_rgb": [float(x) for x in avg_rgb.tolist()],
        "shade_info": {}
    }
    # copy commonly used fields if present
    if isinstance(shade_info, dict):
        for key in ["brand","product","URL","hex_values","hex","Shade Name","shade_name"]:
            if key in shade_info:
                out["shade_info"][key] = shade_info[key]
    # ensure json-safe
    print(json.dumps(out))

if __name__ == "__main__":
    main()
