# Road Pothole Detection System
# A beginner-friendly computer vision project using Streamlit, YOLOv8, and OpenCV.

import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Road Pothole Detection System",
    page_icon="🚗",
    layout="wide"
)

# Application Header & Description
st.title("🚗 Road Pothole Detection System")
st.write(
    "Upload a road image below to detect potholes using a deep learning YOLOv8 model. "
    "The system automatically identifies potholes and smartly recommends whether patch repair "
    "or full road resurfacing is required based on maintenance thresholds."
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider(
    label="Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help="Higher values detect only high-confidence potholes; lower values detect more potential potholes."
)

st.sidebar.header("Road Maintenance Settings")
repair_threshold = st.sidebar.slider(
    label="Pothole Threshold for Full Repair",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="If detected potholes are below this number, patch repair is recommended. If detected potholes reach or exceed this threshold, full road resurfacing is recommended."
)

# ---------------------------------------------------------
# Model Loading Function (Cached for fast performance)
# ---------------------------------------------------------
@st.cache_resource
def load_pothole_model(model_path):
    """
    Loads the YOLOv8 model from disk.
    Cached using @st.cache_resource so the model loads only once in memory.
    """
    if not os.path.exists(model_path):
        return None
    # Load and return the trained YOLO model
    return YOLO(model_path)


# Path to our trained pothole detection weights
MODEL_FILE_PATH = "models/best.pt"

# Verify model file presence before proceeding
if not os.path.exists(MODEL_FILE_PATH):
    st.error(
        f"Model file '{MODEL_FILE_PATH}' was not found! "
        "Please place the trained 'best.pt' file in the 'models/' folder."
    )
    st.stop()

# Load the YOLO model
pothole_model = load_pothole_model(MODEL_FILE_PATH)

# ---------------------------------------------------------
# File Uploader (Accepts static images only)
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    label="Upload a Road Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# ---------------------------------------------------------
# Detection & Display Pipeline
# ---------------------------------------------------------
if uploaded_file is not None:
    # Step 1: Read raw bytes from the uploaded file buffer
    raw_file_bytes = uploaded_file.read()

    # Step 2: Convert raw bytes to a 1D NumPy array of unsigned 8-bit integers
    np_byte_array = np.frombuffer(raw_file_bytes, dtype=np.uint8)

    # Step 3: Decode the byte array into an OpenCV BGR image matrix
    original_bgr = cv2.imdecode(np_byte_array, cv2.IMREAD_COLOR)

    # Validate that image decoding succeeded
    if original_bgr is None:
        st.error("Invalid or corrupted image file. Please upload a valid image.")
        st.stop()

    # Get image dimensions to compute damaged surface area percentage
    image_height, image_width, _ = original_bgr.shape
    total_image_pixels = float(image_height * image_width)
    total_pothole_pixels = 0.0

    # Step 4: Run YOLO inference on the BGR image
    results = pothole_model.predict(
        source=original_bgr,
        conf=confidence_threshold,
        verbose=False
    )

    # Make a copy of the original image to draw detections on
    processed_bgr = original_bgr.copy()

    # List to store confidence scores of all detected potholes
    confidence_list = []

    # Step 5: Extract detected bounding boxes and draw annotations
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            # Extract box coordinates as integers (x1, y1: top-left; x2, y2: bottom-right)
            coords = box.xyxy[0].tolist()
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

            # Accumulate bounding box area for damage ratio calculation
            box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            total_pothole_pixels += box_area

            # Extract confidence score as percentage
            score = float(box.conf[0]) * 100.0
            confidence_list.append(score)

            # Draw green bounding box rectangle on the image
            cv2.rectangle(
                img=processed_bgr,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 255, 0),
                thickness=3
            )

            # Draw text label with confidence percentage above the box
            label_text = f"Pothole: {score:.1f}%"
            label_y = max(y1 - 10, 20)
            cv2.putText(
                img=processed_bgr,
                text=label_text,
                org=(x1, label_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=2
            )

    # Calculate estimated road surface damage percentage
    damage_percentage = (total_pothole_pixels / total_image_pixels) * 100.0

    # Step 6: Convert images from OpenCV BGR format to RGB for Streamlit display
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    # Step 7: Display original vs. processed images side-by-side
    st.subheader("Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_rgb, caption="Original Road Image", use_container_width=True)

    with col2:
        st.image(processed_rgb, caption="Detected Road Image", use_container_width=True)

    # Step 8: Display Summary Metrics
    pothole_count = len(confidence_list)
    st.subheader("Detection Metrics")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Detected Potholes", value=pothole_count)
    with metric_col2:
        st.metric(label="Est. Surface Damage", value=f"{damage_percentage:.2f}%")
    with metric_col3:
        if pothole_count == 0:
            st.metric(label="Recommended Action", value="None Needed")
        elif pothole_count < repair_threshold:
            st.metric(label="Recommended Action", value="Patch Repair")
        else:
            st.metric(label="Recommended Action", value="Full Resurfacing")

    # Step 9: Smart Road Maintenance Recommendation (Based on Threshold)
    st.subheader("🛠️ Smart Road Maintenance Recommendation")

    if pothole_count == 0:
        st.success("🟢 **Road Condition: Good (No Repair Needed)**")
        st.info("No potholes were detected at the current confidence threshold. Routine inspection is sufficient.")
    elif pothole_count < repair_threshold:
        st.warning(f"🟡 **Road Condition: Moderate Damage (Patch Repair Recommended)**")
        st.write(
            f"Detected **{pothole_count}** pothole(s), which is less than the full repair threshold of **{repair_threshold}**. "
            "Isolated potholes can be repaired efficiently using localized asphalt patch filling."
        )
    else:
        st.error(f"🔴 **Road Condition: Critical Damage (Full Road Repair Required)**")
        st.write(
            f"Detected **{pothole_count}** pothole(s), meeting or exceeding the threshold of **{repair_threshold}**. "
            "High pothole frequency indicates deep structural road base fatigue. Full road resurfacing (repaving) is strongly advised for public safety."
        )

    # Step 10: Detection Details List
    if pothole_count > 0:
        st.write("### Pothole Details:")
        for index, score in enumerate(confidence_list, start=1):
            st.write(f"- **Pothole #{index}:** {score:.2f}% confidence")
