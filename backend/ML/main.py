# main.py

# Import the necessary classes and functions from your other scripts
from preprocessing import load_and_preprocess
from feature_extraction import SkinToneAnalyzer
from shade_matching import find_and_recommend_shades

def run_foundation_finder(image_path, dataset_path):
    """
    Runs the complete foundation finder pipeline.
    """
    print("🚀 Starting Foundation Finder...")
    
    try:
        # --- Stage 1: Preprocessing ---
        print("\n--- Stage 1: Preprocessing Image ---")
        preprocessed_image = load_and_preprocess(image_path, do_white_balance=True)
        print("✅ Image preprocessed successfully.")

        # --- Stage 2: Feature Extraction ---
        print("\n--- Stage 2: Analyzing Skin Tone ---")
        analyzer = SkinToneAnalyzer()
        results = analyzer.analyze(preprocessed_image)
        analyzer.close()

        if not results:
            print("\n❌ Could not complete analysis. Please try a different image.")
            return

        hex_code = results['hex_code']
        confidence = results['confidence']

        print(f"\n✅ Analysis Complete!")
        # This is the corrected line
        print(f"   Confidence Score: {confidence}")
        print(f"   Dominant Skin Tone (HEX): {hex_code}")
        
        if confidence == "Low":
            print("\n⚠️ Warning: Confidence is low. Lighting may be uneven. Results may be less accurate.")

        # --- Stage 3: Shade Matching ---
        print("\n--- Stage 3: Finding Foundation Matches ---")
        find_and_recommend_shades(hex_code, dataset_path)

    except FileNotFoundError:
        print(f"❌ Error: Input image not found at '{image_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- CONFIGURE AND RUN YOUR PROJECT HERE ---
    
    # 1. Set the path to the user's input image
    user_image_path = r"C:\Users\shanm\OneDrive\Documents\TARP_proj\user1.jpg"
    
    # 2. Set the path to your foundations dataset
    foundation_dataset_path = 'dataset_preprocessed.csv'
    
    # 3. Run the full process
    run_foundation_finder(user_image_path, foundation_dataset_path)