# ml/feature_extraction.py
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from preprocessing import load_and_preprocess, remove_specular_pixels

# Helper function to display images consistently
def show_image(title, image, wait_key=True):
    """Displays an image window and waits for a key press."""
    # Convert RGB to BGR for OpenCV display
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, bgr_image)
    
    if wait_key:
        print(f"Showing '{title}'. Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(title)

class SkinToneAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.ROI_INDICES = {
            'forehead': [104, 69, 108, 151, 337, 299, 333, 9],
            'left_cheek': [132, 127, 215, 205, 147, 187, 213],
            'right_cheek': [361, 356, 435, 425, 376, 411, 433]
        }

    def analyze(self, image_rgb):
        """
        Processes the image and returns the hex code and visualization images.
        """
        h, w, _ = image_rgb.shape
        visualizations = {}

        # 1. Feature Extraction (MediaPipe Landmarks)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("❌ Warning: No face detected in the image.")
            return None
        
        face_landmarks = results.multi_face_landmarks[0]

        # --- Create Landmark Visualization ---
        landmarks_viz_img = image_rgb.copy()
        self.mp_drawing.draw_landmarks(
            image=landmarks_viz_img,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )
        visualizations['landmarks'] = landmarks_viz_img

        # 2. ROI Identification
        all_skin_pixels = []
        roi_viz_img = image_rgb.copy()
        roi_colors = {'forehead': (255, 0, 0), 'left_cheek': (0, 0, 255), 'right_cheek': (255, 255, 0)} # Red, Blue, Yellow
        
        ## NEW ## Dictionary to hold the dominant color of each region for confidence calculation
        region_dominant_colors = {}

        for region, indices in self.ROI_INDICES.items():
            points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] for i in indices])

            # --- Draw ROIs for Visualization ---
            cv2.polylines(roi_viz_img, [points], isClosed=True, color=roi_colors[region], thickness=2)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, points, 255)
            
            non_specular_mask = remove_specular_pixels(image_rgb)
            final_mask = cv2.bitwise_and(mask, mask, mask=non_specular_mask)
            
            region_pixels = image_rgb[final_mask == 255]
            
            if len(region_pixels) > 0:
                all_skin_pixels.extend(region_pixels)

                ## NEW ## Find the dominant color for this specific region and store it
                if len(region_pixels) > 10: # Ensure enough pixels for clustering
                    kmeans_region = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(region_pixels)
                    unique_labels_region, counts_region = np.unique(kmeans_region.labels_, return_counts=True)
                    dominant_color_region = kmeans_region.cluster_centers_[unique_labels_region[np.argmax(counts_region)]]
                    region_dominant_colors[region] = dominant_color_region

        visualizations['roi'] = roi_viz_img

        if not all_skin_pixels:
            print("❌ Warning: Could not extract sufficient skin pixels.")
            return None

        ## NEW ## Calculate the confidence score by comparing region colors
        confidence = "Low"
        if len(region_dominant_colors) >= 2:
            colors = list(region_dominant_colors.values())
            max_dist = 0
            
            # Calculate Euclidean distance in RGB space between colors
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    dist = np.linalg.norm(colors[i] - colors[j])
                    if dist > max_dist:
                        max_dist = dist
            
            # Set confidence based on the maximum distance
            if max_dist < 35: # Colors are very similar
                confidence = "High"
            elif max_dist < 70: # Colors have some variance
                confidence = "Medium"
            # Otherwise, confidence remains "Low"

        # 3. K-Means Clustering (This part remains unchanged for the final color)
        pixels = np.array(all_skin_pixels)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(pixels)

        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_center = kmeans.cluster_centers_[unique_labels[np.argmax(counts)]]

        # --- Create K-Means Visualization ---
        palette = np.zeros((100, 300, 3), np.uint8)
        dominant_color_rgb = tuple(map(int, dominant_cluster_center))
        
        for i, color in enumerate(kmeans.cluster_centers_):
            color_int = tuple(map(int, color))
            cv2.rectangle(palette, (i * 100, 0), ((i + 1) * 100, 100), color_int, -1)
            
            # Put a star on the dominant color
            if np.array_equal(color, dominant_cluster_center):
                cv2.putText(palette, '*', (i * 100 + 40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        
        visualizations['kmeans'] = palette

        # 4. Final Hex Code
        hex_code = '#%02x%02x%02x' % dominant_color_rgb

        ## MODIFIED ## Add the new confidence score to the return dictionary
        return {
            'hex_code': hex_code,
            'confidence': confidence,
            'visualizations': visualizations
        }

    def close(self):
        self.face_mesh.close()

if __name__ == '__main__':
    try:
        img_path = r"C:\Users\shanm\OneDrive\Documents\TARP_proj\user1.jpg"
        
        # Stage 1: Preprocessing
        print("--- Stage 1: Preprocessing ---")
        preprocessed_image = load_and_preprocess(img_path)
        show_image("1. Preprocessed Image", preprocessed_image)
        
        analyzer = SkinToneAnalyzer()
        results = analyzer.analyze(preprocessed_image)
        analyzer.close()
        
        if results:
            # Stage 2: Feature Extraction (Landmarks)
            print("\n--- Stage 2: Feature Extraction (Landmarks) ---")
            show_image("2. Face Landmarks", results['visualizations']['landmarks'])
            
            # Stage 3: ROI Identification
            print("\n--- Stage 3: ROI Identification ---")
            show_image("3. Regions of Interest", results['visualizations']['roi'])
            
            # Stage 4: K-Means Clustering
            print("\n--- Stage 4: K-Means Clustering ---")
            show_image("4. K-Means Palette (Dominant color marked with *)", results['visualizations']['kmeans'])

            # Final Result
            ## MODIFIED ## Print the new confidence score
            print(f"\n✅ Analysis Complete!")
            print(f" Confidence Score: {results['confidence']}")
            print(f" Dominant Skin Tone (HEX): {results['hex_code']}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
