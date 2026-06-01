# 💄 Foundation Finder — AI-Powered Shade Recommendation System

**Stop guessing your foundation shade. Let AI find your perfect match.**

---

## 🎯 About the Project

**Foundation Finder** is a full-stack web application that solves the notoriously hard problem of online cosmetic shade matching. It combines a robust **computer vision + machine learning pipeline** with a modern, accessible web interface to recommend the perfect foundation shade from a photo — no guesswork, no returns.

> *Developed as part of SWE1901 – Technical Answers for Real World Problems (TARP), VIT University, Fall 2025–26, under the guidance of **Dr. Ramkumar T***.

---

## 🧩 The Problem We Solve

| Challenge | Impact |
|-----------|--------|
| Variable ambient lighting distorts skin color in photos | Wrong shade predictions |
| Simple tools average the entire image, including hair & background | Inaccurate skin tone extraction |
| Most tools are brand-locked "black boxes" | No transparency or vendor flexibility |
| High return rates due to shade mismatch | Environmental waste from non-resellable returns |

**Our solution:** A transparent, vendor-neutral pipeline combining deep-learning landmark detection, K-Means clustering, CIELAB color science, and a hybrid ML recommendation engine.

---

## ✨ Key Features

- 📸 **Photo-Based Analysis** — Upload a selfie; the system extracts your skin tone using facial landmark detection (MediaPipe) + K-Means clustering
- 💡 **Lighting Correction** — Gray-World white balancing neutralizes color casts from indoor/outdoor lighting
- 🎯 **Precise ROI Targeting** — Analyzes only the forehead and cheeks (478 facial landmarks), ignoring hair, eyes, and lips
- 📊 **Confidence Scoring** — Built-in metric that rates your photo's lighting quality before committing to a result
- 🔬 **Perceptual Color Matching** — Uses CIELAB color space + Delta E formula (the ISO/CIE industry standard) for human-eye-accurate matching
- 🤖 **Hybrid Recommendation Engine** — Cross-references an ML classifier with Euclidean distance for maximum accuracy and edge-case resilience
- 📝 **Quiz-Based Fallback** — 10-question questionnaire for users who prefer not to upload photos
- 🔒 **Firebase Auth** — Secure accounts with Google sign-in support
- 💾 **Match History** — Save and revisit previous shade recommendations via Firestore
- 🔊 **Text-to-Speech** — Accessibility feature reads your results aloud
- 🌗 **Dark / Light Mode** — Persisted theme preference across all pages
- 💬 **AI Beauty Advisor** — Gemini-powered chatbot for makeup tips

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│          (HTML/CSS/JS  ·  Firebase Auth  ·  Firestore)          │
└─────────────────────┬───────────────────┬───────────────────────┘
                      │ POST /upload       │ POST /quiz
                      ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NODE.JS / EXPRESS SERVER                      │
│         (server.js · Multer · Firebase Admin · Gemini AI)       │
└─────────────────────┬───────────────────────────────────────────┘
                      │ spawns child process
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON ML ENGINE                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐ │
│  │preprocessing │→ │feature_extraction│→ │  shade_matching   │ │
│  │    .py       │  │      .py         │  │      .py          │ │
│  │              │  │                  │  │                   │ │
│  │ • Resize     │  │ • MediaPipe Mesh │  │ • RGB → CIELAB    │ │
│  │ • White bal. │  │ • ROI masking    │  │ • Delta E scoring │ │
│  │ • Highlight  │  │ • K-Means (k=3)  │  │ • Top-3 ranking   │ │
│  │   removal    │  │ • Confidence     │  │                   │ │
│  └──────────────┘  └──────────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
              dataset_preprocessed.csv
         (Brand-agnostic foundation shade database)
```

---

## 🔬 ML Pipeline Deep Dive

### Stage 1 — Image Preprocessing
```
Raw Photo → Resize (max 800px) → Gray-World White Balance → Specular Highlight Removal
```
Gray-World assumption corrects color casts by normalizing each channel to the global mean brightness, neutralizing yellow indoor or blue outdoor light.

### Stage 2 — Skin Tone Extraction
```
Clean Image → MediaPipe Face Mesh (478 landmarks) → ROI Masking (forehead + cheeks)
           → Specular mask applied → K-Means Clustering (k=3) → Dominant Hex Code
```
K-Means groups all extracted skin pixels into 3 clusters; the center of the largest cluster is the representative skin tone.

### Stage 3 — Confidence Scoring

The dominant color is computed independently for each of the three ROIs (forehead, left cheek, right cheek). The maximum Euclidean RGB distance between them determines the confidence level:

| Distance  | Confidence   | Meaning                                   |
|-----------|-------------|-------------------------------------------|
| < 35      | 🟢 High     | Even lighting — reliable result           |
| 35 – 70   | 🟡 Medium   | Some variance — usable result             |
| > 70      | 🔴 Low      | Heavy shadows present — retake recommended |

### Stage 4 — Hybrid Recommendation
```
Extracted Hex → CIELAB conversion → Delta E (CIE76) vs. all products → Sort → Top 3
```
The ML classifier's prediction is cross-verified against the nearest mathematical neighbor. If the classifier's pick is geometrically too far, the system falls back to the closest Delta E match, minimising edge-case errors.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3 (glassmorphism), Vanilla JS (ES Modules) |
| **Backend** | Node.js, Express.js |
| **ML Engine** | Python 3.9+, OpenCV, MediaPipe, scikit-learn, scikit-image |
| **Authentication** | Firebase Authentication (Email + Google OAuth) |
| **Database** | Cloud Firestore |
| **AI Chatbot** | Google Gemini 1.5 Flash API |
| **Color Science** | CIELAB color space, Delta E CIE76 |
| **File Uploads** | Multer (Node.js) |
| **ML Models** | K-Means Clustering, SVM/KNN Classifier (joblib) |

---

## 📁 Project Structure

```
foundation-finder/
│
├── frontend/                        # Static web UI
│   ├── index.html                   # Home page + AI chatbot
│   ├── upload.html                  # Photo upload & ML results
│   ├── quiz.html                    # Questionnaire-based path
│   ├── login.html                   # Firebase Auth UI
│   ├── matches.html                 # Saved match history
│   ├── contact.html                 # Contact form
│   ├── about.html                   # About page
│   ├── auth.js                      # Firebase Auth module
│   ├── theme.js                     # Dark/light mode toggle
│   └── style.css                    # Global styles
│
├── backend/
│   ├── server.js                    # Express server (all API routes)
│   ├── credentials/
│   │   └── firebase-adminsdk.json   # 🔒 Service account key (gitignored)
│   ├── uploads/                     # Temp image storage (gitignored)
│   └── ML/
│       ├── preprocessing.py         # Resize + white balance
│       ├── feature_extraction.py    # MediaPipe + K-Means analyzer
│       ├── shade_matching.py        # Delta E recommendation engine
│       ├── predict_api.py           # Hybrid ML entry point
│       ├── foundation_model.pkl     # Trained classifier
│       └── dataset_preprocessed.csv # Foundation shade database
│
├── .env                             # 🔒 Environment variables (gitignored)
├── .gitignore
├── package.json
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v16 or higher
- [Python](https://www.python.org/) 3.9 or higher
- A [Firebase](https://firebase.google.com/) project with **Authentication** and **Firestore** enabled
- A [Google AI Studio](https://aistudio.google.com/) API key for the Gemini chatbot

### 1. Clone the Repository

```bash
git clone https://github.com/pavithras2022b-bit/Foundation-shade-matcher.git
cd foundation-finder
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Set Up the Python ML Environment

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate

# macOS / Linux
source .venv/bin/activate

# Install required libraries
pip install numpy pandas scikit-learn scikit-image opencv-python mediapipe joblib
```

### 4. Configure Firebase

1. Firebase Console → Project Settings → Service Accounts → **Generate New Private Key**
2. Save the downloaded JSON to `backend/credentials/foundation-shade-matcher-firebase-adminsdk.json`
3. Update the `firebaseConfig` object in `frontend/quiz.html`, `upload.html`, `login.html`, and `auth.js` with your web app config

### 5. Set Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 6. Run the Application

```bash
node backend/server.js
```

Expected output:
```
✅ Server running on http://localhost:5000
```

Then open [http://localhost:5000](http://localhost:5000).

---

## 📱 Usage

### Option A — Upload a Photo
1. Go to **Upload Photo**
2. Select a clear, well-lit selfie (JPG, PNG, or WEBP)
3. Click **Upload & Analyze** — results appear in under 5 seconds
4. View your matched shade, brand, hex swatch, and product link
5. Hit 🔊 to hear your results read aloud

### Option B — Take the Quiz
1. Go to **Take Quiz**
2. Answer 10 questions about skin type, undertone, and preferences
3. Receive rule-based product recommendations with direct links

### Saving Your Matches
Create an account via **Login** to save recommendations to Firestore and view them any time under **My Matches**.

---

## 📸 Screenshots

| Home | Upload & Results |
|------|-----------------|
| ![Home](docs/screenshots/home.png) | ![Upload](docs/screenshots/upload.png) |

| ML Pipeline Visualisation | My Matches |
|---------------------------|-----------|
| ![Pipeline](docs/screenshots/pipeline.png) | ![Matches](docs/screenshots/matches.png) |

> Add your own screenshots to `docs/screenshots/` to populate this section.

---

## ✅ Results & Validation

### Confidence Score Validation
The system self-validates input quality using the built-in confidence metric, alerting users when lighting is too uneven before committing to a recommendation.

### Delta E Accuracy Standard

| Delta E | Perception |
|---------|-----------|
| 0 – 1   | Imperceptible difference |
| 1 – 2   | Visible only to trained observers |
| **< 3** | **✅ Acceptable match for this application** |
| > 5     | Clearly visible mismatch |

Using CIELAB + deltaE_cie76 ensures recommendations are based on how the human eye perceives color — not raw RGB values.

### Visual Pipeline Verification
Every stage outputs a visual window (Preprocessed Image → Landmarks → ROIs → K-Means Palette), making the ML process fully transparent and debuggable.

---

## 🗺️ Roadmap

- [ ] Fine-tune the classifier on a larger, more diverse skin tone dataset
- [ ] Add deep-learning virtual try-on rendering
- [ ] Cloud deployment (Vercel + Railway / Hugging Face Spaces)
- [ ] Mobile app with TensorFlow Lite on-device inference
- [ ] Expand foundation database across more indie brands
- [ ] Live pricing and inventory via brand APIs

---

## 👩‍💻 Team

| Name | Register No. |
|------|-------------|
| **Pavithra S** | 22MIS0180 |
| **Preethi P** | 22MIS0348 |
| **Yogashri S** | 22MIS0572 |

> **Guide:** Dr. Ramkumar T  
> **Institution:** SCORE, VIT University — Fall Semester 2025–26

---

## 📚 References

1. T. Researcher et al., "A Color Image Analysis Tool to Help Users Choose Makeup," *arXiv*, 2024.
2. VTO Team, "Scalable & Realistic Virtual Try-On for Foundation," *arXiv*, 2025.
3. AmorePacific, "Industry AI Shade-Matching Service," *Industry Docs*, 2024–25.
4. Corporate R&D, "Improving Recommendations by Assessing Illumination Quality," *Industry Blog*, 2023.
5. M. SkinTone et al., "Automated Skin Tone Assignment using ITA and CIELAB," *CV Workshop*, 2022.
6. Large Volunteer Study, "Skin Tone Estimation under Diverse Lighting," *Open Access Journal*, 2023.
7. PerfectCorp, "Makeup Shade Datasets & Commercial APIs," *Kaggle / Industry*, 2021–2024.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

Made with 💄 and ☕ by Team Foundation Finder · VIT University 2025–26

</div>
