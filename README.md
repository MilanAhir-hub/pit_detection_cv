# 🚗 Road Pothole Detection System

A simple, lightweight, and deployment-ready Computer Vision web application built with **Streamlit**, **Ultralytics YOLOv8**, and **OpenCV** to detect potholes in road images and report confidence metrics.

---

## 📌 Project Overview
Potholes pose major risks to vehicle safety, infrastructure longevity, and road maintenance planning. This project provides an accessible web interface where users can upload road photographs, adjust sensitivity settings, and instantly inspect detected potholes with visual bounding boxes, confidence percentages, and total detection metrics.

### Whitelisted Core Technologies
| Technology | Role in This Project | Why We Need It |
| :--- | :--- | :--- |
| **Python** | Core Language | Glue between the user interface, image processing, and deep learning model. |
| **Ultralytics YOLOv8** | Object Detection AI | Single-stage neural network that predicts bounding boxes (`x1, y1, x2, y2`) and confidence scores. |
| **OpenCV (`cv2`)** | Computer Vision / Drawing | Decodes image byte streams, draws bounding boxes and text labels, and converts color spaces. |
| **Streamlit** | Web Interface | Pure Python frontend providing interactive sliders, file uploaders, and side-by-side comparison columns. |
| **NumPy** | Numerical Array Processing | Converts raw upload byte streams into 1D `uint8` arrays suitable for OpenCV image decoding. |

---

## 📂 Project Structure
```text
road-pothole-detection/
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Python package dependencies
├── packages.txt            # System apt dependencies for Streamlit Cloud
├── planning.md             # Project design & roadmap specification
├── README.md               # Documentation and College Viva guide
├── models/
│   └── best.pt             # Pretrained YOLOv8s pothole detection weights
└── sample_images/
    ├── road_with_pothole.jpg   # Road image containing potholes
    └── clean_road.jpg          # Clear road image without potholes
```

---

## 🚀 Local Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone <your-repo-url>
cd "road-pothole-detection"
```

### 2. Create and Activate a Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS / Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)
Deploying to **Streamlit Community Cloud** takes 3 simple steps:

1. **Push Repository to GitHub:**
   Ensure all files (`app.py`, `requirements.txt`, `packages.txt`, and `models/best.pt`) are committed to a GitHub repository.
2. **Connect to Streamlit Cloud:**
   Log in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. **Deploy App:**
   - Click **"New app"**.
   - Select your repository and branch (`main`).
   - Set **Main file path** to `app.py`.
   - Click **"Deploy!"**.
   *(The included `packages.txt` ensures system-level OpenGL libraries install automatically on the Linux container).*

---

## 🎓 College Viva / Interview Q&A

### Q1: Why use YOLOv8 instead of traditional CNNs or older YOLO versions?
**Answer:**
Traditional Convolutional Neural Networks (like AlexNet or VGG) perform image classification (labeling the whole image) rather than object detection (locating where objects are). Two-stage detectors (like Faster R-CNN) are computationally heavy and slow. YOLOv8 is an anchor-free, single-stage detector that predicts bounding boxes and class probabilities in a single forward pass, providing high accuracy (`mAP`) and real-time inference speeds suitable for practical deployments.

### Q2: Why do we convert images with `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`?
**Answer:**
OpenCV historically reads and decodes images in **BGR** (Blue-Green-Red) channel order, whereas web browsers, HTML canvas elements, and Streamlit display images using the standard **RGB** (Red-Green-Blue) color model. If we do not convert BGR to RGB before rendering, the asphalt road and surrounding scenery would appear with an incorrect inverted blue/orange tint.

### Q3: What is the purpose of the Confidence Threshold slider?
**Answer:**
The model produces a confidence score ($0.0$ to $1.0$) for every potential detection candidate. The confidence slider filters out predictions below the selected cutoff. Setting a higher threshold (e.g., $0.50$) suppresses weak detections and prevents false positives (such as dark tree shadows or wet patches). Setting a lower threshold (e.g., $0.20$) helps detect faint, distant, or partially occluded potholes.

### Q4: Why is `opencv-python-headless` used instead of `opencv-python`?
**Answer:**
Standard `opencv-python` includes dependencies on X11 and Qt GUI display libraries. Cloud server environments (such as Streamlit Community Cloud or Docker containers) run in headless Linux modes without graphical display servers. Standard OpenCV crashes immediately on import in such environments. `opencv-python-headless` provides all image manipulation and drawing routines without GUI window dependencies.

### Q5: Why is `@st.cache_resource` used for loading the model?
**Answer:**
Streamlit re-executes the Python script from top to bottom whenever the user interacts with an input widget (like moving the confidence slider). Loading the 22.5 MB YOLO model file from disk on every interaction would introduce noticeable latency. Decorating `load_pothole_model()` with `@st.cache_resource` keeps the initialized model in memory across re-runs.

### Q6: How does the Smart Road Maintenance Recommendation work?
**Answer:**
Civil engineering standards distinguish between localized surface distress and systemic structural failure. The system calculates two key metrics:
1. **Pothole Frequency vs. Threshold:** If the number of potholes is below the user-configurable threshold (default: 3), spot **Patch Repair** is advised as a cost-effective localized fix. If the count meets or exceeds the threshold, the road base has likely suffered severe degradation, making **Full Road Resurfacing** mandatory.
2. **Estimated Surface Damage Ratio:** Computes total bounding box pixels divided by total road image pixels ($\frac{\sum \text{Box Area}}{\text{Image Area}} \times 100\%$) to quantify surface deterioration percentage.

---

## ⚠️ System Limitations
1. **2D Static Analysis:** The application detects pothole surface area from 2D images but does not measure 3D pothole depth (in centimeters) or road gradient.
2. **Visual Ambiguity:** Strong contrast conditions (such as deep tree shadows, oil stains, or dark manhole covers) can occasionally trigger false positives at very low confidence thresholds.
3. **Lighting & Weather:** Nighttime photography and heavy rainfall reduce camera clarity and model detection confidence compared to daylight scenes.

