# ml/preprocessing.py
import cv2
import numpy as np

def load_and_preprocess(image_path, target_long=800, do_white_balance=True):
    """
    Load image, convert BGR->RGB, resize (keeping aspect ratio), optional gray-world white balance.
    Returns: RGB uint8 image (H, W, 3)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # BGR -> RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize: keep aspect ratio, scale longest side to target_long
    h, w = img.shape[:2]
    if max(h, w) != target_long:
        if h >= w:
            new_h = target_long
            new_w = int(w * (target_long / h))
        else:
            new_w = target_long
            new_h = int(h * (target_long / w))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if do_white_balance:
        img = gray_world_white_balance(img)

    return img

def gray_world_white_balance(img_rgb):
    """
    Simple Gray-World white balance. Input: RGB uint8, output: RGB uint8.
    """
    img = img_rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    # avoid division by zero
    means = np.maximum(means, 1e-6)
    mean_gray = means.mean()
    scale = mean_gray / means
    img_bal = img * scale
    img_bal = np.clip(img_bal, 0, 255).astype(np.uint8)
    return img_bal

def remove_specular_pixels(img_rgb, v_threshold=0.92):
    """
    Returns a uint8 mask (255 = keep) that removes very bright specular pixels.
    img_rgb in uint8 [0,255].
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    v = img_hsv[..., 2]
    mask = v <= v_threshold
    # Convert boolean (True/False) to uint8 (255/0) for OpenCV
    return mask.astype(np.uint8) * 255

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = r"C:\Users\shanm\OneDrive\Documents\TARP_proj\user1.jpg"  # <-- put any sample image path here

    processed_img = load_and_preprocess(img_path)
    plt.imshow(processed_img)
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()

