import pandas as pd
from skimage.color import rgb2lab, deltaE_cie76
import numpy as np

def hex_to_rgb_array(hex_code):
    """Converts a hex color string to a NumPy array of RGB values."""
    hex_clean = hex_code.lstrip('#')
    return np.array([int(hex_clean[i:i+2], 16) for i in (0, 2, 4)])

def find_and_recommend_shades(user_hex, dataset_path, top_n=3):
    """
    Loads the dataset, finds the top N closest shades, and prints the recommendations.

    Args:
        user_hex (str): The hex code of the user's skin tone.
        dataset_path (str): The file path to the foundation CSV dataset.
        top_n (int): The number of recommendations to return.
    """
    try:
        # 1. Load your foundation dataset.
        foundations_df = pd.read_csv(dataset_path)

        # 2. Convert user's hex to the LAB color space for accurate comparison.
        user_rgb = hex_to_rgb_array(user_hex)
        user_lab = rgb2lab(np.array([[user_rgb / 255.0]], dtype=float))

        # 3. Calculate the color difference (Delta E) for each foundation.
        color_differences = []
        for index, row in foundations_df.iterrows():
            foundation_hex = row['Hex']
            foundation_rgb = hex_to_rgb_array(foundation_hex)
            foundation_lab = rgb2lab(np.array([[foundation_rgb / 255.0]], dtype=float))
            
            delta_e = deltaE_cie76(user_lab, foundation_lab)[0,0]
            color_differences.append(delta_e)
        
        foundations_df['delta_e'] = color_differences

        # 4. Sort by the smallest difference to find the best matches.
        closest_matches = foundations_df.sort_values(by='delta_e').head(top_n)

        # 5. Display the recommendations.
        print("\n🎉 Here are your Top 3 Foundation Recommendations!")
        print("--------------------------------------------------")
        recommendations = closest_matches.reset_index(drop=True)
        for index, row in recommendations.iterrows():
            print(f"  #{index + 1}:")
            print(f"    Brand:     {row['Brand']}")
            print(f"    Product:   {row['Product']}")
            print(f"    Shade:     {row['Shade Name']}")
            print(f"    Hex:       {row['Hex']}")
            print(f"    Match Score (ΔE): {row['delta_e']:.2f} (Lower is better)") 
        print("--------------------------------------------------")

    except FileNotFoundError:
        print(f"❌ Error: The dataset file '{dataset_path}' was not found.")
    except Exception as e:
        print(f"An error occurred in the matching process: {e}")

# This block is for testing this file directly if needed.
if _name_ == '_main_':
    # Example usage for testing:
    test_hex = '#968477'
    test_dataset = 'dataset_preprocessed.csv'
    print(f"--- Running standalone test for shade_matching.py ---")
    find_and_recommend_shades(test_hex, test_dataset)