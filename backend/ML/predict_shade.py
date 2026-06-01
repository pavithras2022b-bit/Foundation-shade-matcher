# ML/predict_shade.py
import cv2
import numpy as np
import joblib
import os
from sklearn.metrics import pairwise_distances_argmin_min

def extract_skin_rgb(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"⚠️ Could not read image at {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower, upper)

    skin = cv2.bitwise_and(image_rgb, image_rgb, mask=skin_mask)
    skin_pixels = skin[skin_mask > 0]

    if len(skin_pixels) == 0:
        raise ValueError("⚠️ No skin detected in image")

    avg_rgb = np.mean(skin_pixels, axis=0)
    return avg_rgb

def recommend_foundation(image_path, model_path="ML/foundation_model.pkl"):
    model_data = joblib.load(model_path)
    scaler = model_data["scaler"]
    model = model_data["model"]
    data = model_data["data"]

    avg_rgb = extract_skin_rgb(image_path)
    avg_rgb_scaled = scaler.transform([avg_rgb])

    ml_pred = model.predict(avg_rgb_scaled)[0]
    dataset_rgbs = data[["R", "G", "B"]].values
    idx, _ = pairwise_distances_argmin_min([avg_rgb], dataset_rgbs)
    rule_pred = data.iloc[idx[0]]["shade_name"]

    # Hybrid approach
    if ml_pred == rule_pred:
        final_shade = ml_pred
    else:
        dist_ml = np.linalg.norm(avg_rgb - np.mean(dataset_rgbs[data["shade_name"] == ml_pred], axis=0))
        dist_rule = np.linalg.norm(avg_rgb - dataset_rgbs[idx[0]])
        final_shade = ml_pred if dist_ml < dist_rule else rule_pred

    shade_info = data[data["shade_name"] == final_shade].iloc[0]
    return {
        "predicted_shade": final_shade,
        "brand": shade_info["brand"],
        "product": shade_info["product"],
        "URL": shade_info["URL"],
        "hex": shade_info["hex_values"],
        "avg_rgb": avg_rgb.tolist()
    }

if __name__ == "__main__":
    test_image = "user1.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️ Test image not found at {test_image}")
    else:
        result = recommend_foundation(test_image)
        print("\n🎨 Recommended Foundation Shade:")
        print(result)

