# ml/model_integration.py
from pre_processing import load_and_preprocess
from feature_extraction import extract_skin_features
from shade_matching import match_skin_shade

def predict_shade(image_path):
    img = load_and_preprocess(image_path)
    features = extract_skin_features(img)
    shade = match_skin_shade(features)
    return shade

if __name__ == "__main__":
    result = predict_shade("test_images/sample.jpg")
    print("Predicted Shade:", result)
