# Project Specification & Agent Execution Plan: Road Pothole Detection System

## 1. Agent Role & Strict Behavioral Rules
You are an AI Coding Agent building a beginner-friendly, solo college project titled **Road Pothole Detection System**. You MUST follow these rules without exception:

1. **Write Code Like a Junior Developer:**
   - Write flat, procedural, easy-to-read Python code inside a single `app.py` file.
   - Use simple functions (maximum 2–3 helper functions) and clear, descriptive variable names (e.g., `uploaded_file`, `pothole_count`, `confidence_score`).
   - Add clear, beginner-friendly comments above key lines so a student can explain every line during a college viva.
   - **FORBIDDEN:** Do NOT use Object-Oriented Programming (no custom `class` definitions), complex design patterns, decorators (except `@st.cache_resource` for loading the model once), complex type annotations, `pathlib` abstractions, or multi-file module imports.
2. **Strict Technology Whitelist:**
   - Use ONLY: **Python**, **Ultralytics YOLO (`ultralytics`)**, **OpenCV (`cv2`)**, **Streamlit (`streamlit`)**, and **NumPy (`numpy`)**.
   - Do NOT import `PIL` / `Pillow`, `matplotlib`, `pandas`, `torch`, or any other third-party library in `app.py`. Decode uploaded bytes strictly using `numpy.frombuffer()` and `cv2.imdecode()`.
3. **Strict Feature Scope (IMAGE INPUT ONLY):**
   - Support static image uploads (`.jpg`, `.jpeg`, `.png`) ONLY.
   - **FORBIDDEN:** No video processing, no live webcam (`st.camera_input`), no GPS/maps, no databases, no login/authentication, no command-line interface (`argparse`), and no custom model training scripts.
4. **Deployment-Ready Architecture:**
   - Make the project 100% ready to deploy later on **Streamlit Community Cloud**.
   - Use `opencv-python-headless` in `requirements.txt` (standard `opencv-python` crashes on Linux cloud servers due to missing X11 display drivers).
   - Include a `packages.txt` file containing `libgl1` and `libglib2.0-0` so Linux containers never fail on OpenCV imports.
   - Use relative file paths only (`models/best.pt`), never hardcoded absolute local paths (`C:\Users\...`).
5. **Task-by-Task Execution & Mandatory Testing:**
   - Execute **one task at a time** from Section 8 below.
   - After completing a task, you MUST run the **Mandatory Test Cases** for that task before moving to the next task. If any test fails, fix the issue first.

---

## 2. System Architecture & End-to-End Workflow
The application follows a strict 7-step pipeline:
1. **Open Web App:** User opens the Streamlit interface (`streamlit run app.py`).
2. **Upload & Configure:** User adjusts a sidebar/main confidence threshold slider (`0.10` to `0.90`, default `0.25`) and uploads a `.jpg`, `.jpeg`, or `.png` road image.
3. **Validate & Decode Image:**
   - Read raw bytes from Streamlit's `UploadedFile`.
   - Convert bytes to a 1D NumPy `uint8` array via `np.frombuffer()`.
   - Decode into a 3-channel BGR OpenCV image matrix via `cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)`.
   - Verify the decoded image is not `None`.
4. **YOLO Model Analysis:**
   - Load the pretrained pothole model (`models/best.pt`) once using `@st.cache_resource`.
   - Run `results = model.predict(source=image_bgr, conf=confidence_threshold, verbose=False)`.
5. **Extract Detections & Draw Bounding Boxes:**
   - Create a copy of the original BGR image (`annotated_bgr = image_bgr.copy()`).
   - Loop through `results[0].boxes`. For each box:
     - Extract coordinates `x1, y1, x2, y2` (converted to integers).
     - Extract the confidence score `conf` (float converted to percentage, e.g., `85.4%`).
     - Draw a visible rectangle on `annotated_bgr` using `cv2.rectangle()`.
     - Draw the label `"Pothole: 85.4%"` above the box using `cv2.putText()`.
6. **Display Original vs. Processed Images:**
   - Convert both `image_bgr` and `annotated_bgr` from BGR to RGB using `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.
   - Display them side-by-side using `st.columns(2)`.
7. **Display Summary & Confidence Scores:**
   - If `pothole_count > 0`, show a warning/metric banner with the total number of detected potholes and list each pothole's confidence score (e.g., `Pothole #1: 86.2% confidence`).
   - If `pothole_count == 0`, display a success message: `"No potholes detected in this road image at the current confidence threshold."`

---

## 3. Pretrained Model Selection & Verification Guide
Do NOT use `yolov8n.pt` or `yolov8s.pt` (those are generic 80-class COCO models that do not know what a pothole is).

- **Recommended Pretrained Sources (Hugging Face):**
  1. **Option A (Primary):** `peterhdd/pothole-detection-yolov8` (`https://huggingface.co/peterhdd/pothole-detection-yolov8`) — YOLOv8s trained for 100 epochs on road potholes, License: **Apache-2.0**, 1 class (`pothole`).
  2. **Option B (Backup):** `Samdutse/pothole-yolov8` (`https://huggingface.co/Samdutse/pothole-yolov8/resolve/main/best.pt`) — YOLOv8s trained on the Smartathon pothole dataset (`mAP50: 0.816`), file named `best.pt`.
- **Where to Place It:** Save the downloaded `.pt` file as `models/best.pt`.
- **How the Agent Must Verify It:**
  - Load the model in Python and inspect `model.names`.
  - Confirm that the class dictionary contains `'pothole'` (case-insensitive) and does NOT contain 80 COCO classes (`'person'`, `'bicycle'`, `'car'`).
  - If `models/best.pt` is missing when the user runs the app, `app.py` must show a clear `st.error()` explaining where to place `models/best.pt` rather than crashing with a raw Python traceback.

---

## 4. Why Each of the 5 Technologies Is Needed (Viva Reference)
| Technology | Role in This Project | Why We Need It |
| :--- | :--- | :--- |
| **Python** | Main language | Connects the UI, image processing, and deep learning library with clean, readable syntax. |
| **YOLO (`ultralytics`)** | Object Detection AI | Loads `models/best.pt` and predicts bounding box coordinates (`x1, y1, x2, y2`) and confidence scores in a single pass. |
| **OpenCV (`cv2`)** | Image Processing | Decodes uploaded image bytes, draws bounding boxes/text on the road image, and converts BGR to RGB for display. |
| **Streamlit** | Web Frontend | Provides the browser UI (`st.file_uploader`, `st.slider`, `st.columns`, `st.image`) in pure Python without HTML/JS. |
| **NumPy** | Array Manipulation | Converts the raw byte stream of an uploaded image into an array of unsigned 8-bit integers (`np.uint8`) that OpenCV can read. |

---

## 5. Minimum Folder Structure
Do NOT create extra folders (`src/`, `utils/`, `components/`, `config/` are forbidden). Keep the exact structure below:

```text
road-pothole-detection/
├── app.py                  # Main Streamlit web application (under 150 lines of clean code)
├── requirements.txt        # Python packages for local run and cloud deployment
├── packages.txt            # Linux apt packages required by OpenCV on Streamlit Cloud
├── planning.md             # This project plan and agent instruction file
├── README.md               # Setup, testing, deployment, and viva guide
├── models/
│   └── best.pt             # Pretrained YOLOv8 pothole detection weights
└── sample_images/
    ├── road_with_pothole.jpg
    └── clean_road.jpg
```

---

## 6. Required Packages & Installation Commands

### `requirements.txt` (Deployment-Ready)
```text
streamlit>=1.38.0
ultralytics>=8.2.0
opencv-python-headless>=4.10.0.84
numpy>=1.26.0
```

### `packages.txt` (For Streamlit Community Cloud Deployment)
```text
libgl1
libglib2.0-0
```

### Local Installation Commands
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 7. Expected Output & Known System Limitations (For College Viva)
- **Expected Output:**
  - Side-by-side comparison of Original Image vs. Detected Image.
  - Bright, easy-to-read bounding boxes with confidence percentages around each pothole.
  - Total pothole counter and a bulleted list of confidence scores for every detection.
- **Known Limitations:**
  1. **Static Images Only:** Does not estimate pothole physical depth (in cm) or vehicle speed impact.
  2. **Visual Lookalikes:** Dark shadows, wet puddles, oil stains, or manhole covers may occasionally trigger false positives at low confidence thresholds.
  3. **Lighting Conditions:** Nighttime or heavy rain images reduce detection confidence compared to clear daylight photos.

---

## 8. Step-by-Step Agent Implementation Roadmap & Mandatory Test Cases

> **AGENT INSTRUCTION:** Execute Tasks 1 through 5 sequentially. After finishing the code/files for a task, immediately execute its **Mandatory Test Cases** in the terminal and confirm they pass before starting the next task.

### Task 1: Project Scaffolding & Deployment Files
- **What to Do:**
  1. Create directories `models/` and `sample_images/`.
  2. Create `requirements.txt` with the exact 4 libraries listed in Section 6.
  3. Create `packages.txt` with `libgl1` and `libglib2.0-0`.
- **Mandatory Test Cases (Run After Task 1):**
  - **Test 1.1:** Verify `models/` and `sample_images/` directories exist.
  - **Test 1.2:** Run `pip install -r requirements.txt` and verify exit code `0`.
  - **Test 1.3:** Run `python -c "import streamlit, ultralytics, cv2, numpy; print('Imports OK')"` and confirm it prints `Imports OK` with zero errors.

### Task 2: Obtain & Verify Pretrained Pothole Model (`models/best.pt`)
- **What to Do:**
  1. Check if `models/best.pt` exists. If not, download the verified YOLOv8 pothole weights from Hugging Face (`https://huggingface.co/Samdutse/pothole-yolov8/resolve/main/best.pt` or `peterhdd/pothole-detection-yolov8`) into `models/best.pt`.
  2. Download 1 sample road pothole image (`sample_images/road_with_pothole.jpg`) and 1 clean road image (`sample_images/clean_road.jpg`) from open Wikimedia Commons / public domain URLs for testing.
- **Mandatory Test Cases (Run After Task 2):**
  - **Test 2.1 (File Check):** Verify `models/best.pt` exists and its file size is greater than 5 MB.
  - **Test 2.2 (Class Verification):** Run:
    ```bash
    python -c "from ultralytics import YOLO; m = YOLO('models/best.pt'); print(m.names); assert len(m.names) <= 5 and 'person' not in [v.lower() for v in m.names.values()], 'ERROR: Generic COCO model detected!'"
    ```
    Confirm the assertion passes and outputs the pothole class dictionary.

### Task 3: Write `app.py` (Junior Developer Style)
- **What to Do:**
  Write `app.py` (around 90–130 lines) using simple procedural structure and clear comments:
  1. Import **only** `os`, `cv2`, `numpy as np`, `streamlit as st`, and `from ultralytics import YOLO`.
  2. Configure the page with `st.set_page_config(page_title="Road Pothole Detection System", layout="wide")`.
  3. Add a title, short description, and a confidence slider:
     `confidence_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)`
  4. Write a simple cached function `load_pothole_model(model_path)` using `@st.cache_resource` that loads `YOLO(model_path)`. Check `os.path.exists("models/best.pt")` first so a friendly error is shown if the file is missing.
  5. Add `uploaded_file = st.file_uploader("Upload a Road Image", type=["jpg", "jpeg", "png"])`.
  6. When an image is uploaded:
     - Convert bytes using `file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)` and `original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)`.
     - Check if `original_bgr is None` and show `st.error("Invalid or corrupted image file.")` if so.
     - Run `results = model.predict(source=original_bgr, conf=confidence_threshold, verbose=False)`.
     - Copy the image (`processed_bgr = original_bgr.copy()`), loop through `results[0].boxes`, extract `x1, y1, x2, y2` and `conf`, draw green or red rectangles (`cv2.rectangle`) and labels (`cv2.putText`), and store confidence scores in a simple Python list `confidence_list`.
     - Convert both images from BGR to RGB using `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` and display them side-by-side in two `st.columns(2)`.
     - Display total potholes detected (`len(confidence_list)`) and loop over `confidence_list` to print each detection's confidence score, or show `st.success("No potholes detected in this image.")` if `len(confidence_list) == 0`.
- **Mandatory Test Cases (Run After Task 3):**
  - **Test 3.1 (Syntax Check):** Run `python -m py_compile app.py` and confirm zero syntax errors.
  - **Test 3.2 (Strict Import Audit):** Inspect imports in `app.py` to confirm no banned libraries (`PIL`, `matplotlib`, `pandas`, etc.) were used.

### Task 4: Headless Detection & Edge-Case Verification
- **What to Do:**
  Test the core detection pipeline on the sample images and edge cases before finalizing documentation.
- **Mandatory Test Cases (Run After Task 4):**
  - **Test 4.1 (Positive Detection Test):** Run a one-line Python verification passing `sample_images/road_with_pothole.jpg` through `YOLO('models/best.pt')` at `conf=0.25` and verify it returns bounding box coordinates and confidence scores without errors.
  - **Test 4.2 (Zero-Detection / High-Threshold Test):** Run the model on a blank/clean image array (`np.zeros((640, 640, 3), dtype=np.uint8)`) and confirm `len(results[0].boxes) == 0` is handled gracefully without index errors.
  - **Test 4.3 (Streamlit Smoke Test):** Launch `streamlit run app.py --server.headless true` for 5 seconds and confirm the server starts cleanly on `http://localhost:8501` with no runtime tracebacks.

### Task 5: Write `README.md` (Viva & Cloud Deployment Guide)
- **What to Do:**
  Create a concise `README.md` explaining:
  1. Project overview and the 5 technologies used.
  2. Local setup and run instructions (`streamlit run app.py`).
  3. How to deploy to **Streamlit Community Cloud** in 3 clicks (Push repo to GitHub -> Connect to `share.streamlit.io` -> Select `app.py` and click Deploy).
  4. Quick Viva Q&A (Why YOLOv8? Why convert BGR to RGB? What does the confidence threshold do?).
- **Mandatory Test Cases (Run After Task 5):**
  - **Test 5.1 (Final File Tree Audit):** Verify the project root contains only the approved files (`app.py`, `requirements.txt`, `packages.txt`, `planning.md`, `README.md`, `models/best.pt`, and `sample_images/`) and no leftover temporary test scripts.

### Task 6: Smart Road Repair Decision & Surface Damage Analysis
- **What to Do:**
  1. Add a sidebar maintenance slider (`repair_threshold`, range 1–10, default 3) defining the threshold count between patch repair and full road repaving.
  2. Compute total road surface damage percentage by summing bounding box areas relative to the image resolution ($\frac{\sum \text{Box Area}}{\text{Image Area}} \times 100\%$).
  3. Display a 3-metric dashboard: Total Potholes, Estimated Surface Damage %, and Recommended Action.
  4. Render color-coded actionable advice:
     - 🟢 **Good Condition (No Repair Needed):** If `pothole_count == 0`.
     - 🟡 **Patch Repair Recommended:** If `0 < pothole_count < repair_threshold`.
     - 🔴 **Full Road Resurfacing Required:** If `pothole_count >= repair_threshold`.
- **Mandatory Test Cases (Run After Task 6):**
  - **Test 6.1 (Syntax & Whitelist Check):** Verify zero syntax errors and 100% adherence to whitelisted libraries (`os`, `cv2`, `numpy`, `streamlit`, `ultralytics`) with 0 custom classes.
  - **Test 6.2 (Boundary Logic Check):** Verify decision thresholds for counts `0`, `2 < 3` (Patch Repair), `3 == 3` (Full Resurfacing), and `5 > 3` (Full Resurfacing).
  - **Test 6.3 (Damage Ratio Bounds):** Verify damage ratio is strictly `0.0%` for clean road and `> 0.0%` and `<= 100.0%` for damaged road.