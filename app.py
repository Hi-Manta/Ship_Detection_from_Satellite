from ultralytics import YOLO
import cv2
import os
import streamlit as st
from datetime import datetime
from PIL import Image
import numpy as np
import tempfile

# ------------------------- YOLO Processing Functions -------------------------
def process_image(model, image_path):
    image = cv2.imread(image_path)
    results = model(image)
    summary = []

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls)
            class_name = model.names[class_id]
            confidence = round(float(conf) * 100, 2)
            length = round(abs(x2 - x1) * 3.1416, 2)
            latitude = round(23.8103 + (y1 * 0.00001), 2)
            longitude = round(90.4125 + (x1 * 0.00001), 2)

            label_lines = [
                f"Object: {class_name}",
                f"Confidence: {confidence}%",
                f"Length: {length} meters",
                f"Coordinates: ({latitude}, {longitude})"
            ]

            text_x = x1
            text_y = y1 - 10 if y1 - 100 > 10 else y2 + 10

            # Apply padding to avoid overlap
            text_padding = 10
            for i, line in enumerate(label_lines):
                cv2.putText(image, line, (text_x + text_padding, text_y + (i * 35)),  # Added padding here
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            summary.append({
                'Object': class_name,
                'Confidence (%)': confidence,
                'Length (m)': length,
                'Latitude': latitude,
                'Longitude': longitude
            })

    return image, summary

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="Ship Detection App", layout="wide")
st.markdown("""
    <style>
    section[data-testid="stSidebar"] .element-container label, 
    section[data-testid="stSidebar"] .stSlider label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;  /* White for better visibility */
    }
    section[data-testid="stSidebar"] {
        background-color: #003366;  /* Dark blue background */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛳️ Ship & Object Detection from Satellite")
st.markdown("""
Welcome to the **Ship Detection App**. Upload an image or video and let the model detect regular ships, cargo ships, islands, and clouds. The app provides coordinates, estimated length, confidence scores, and more.
""")

with st.sidebar:
    st.header("🔧 Settings")
    input_type = st.radio("Select Input Type", ("Image", "Video"))
    uploaded_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg', 'mp4'])
    download_placeholder = st.empty()

if uploaded_file:
    with st.spinner("Processing..."):
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        temp_path = os.path.join(tempfile.gettempdir(), f"temp_input{file_ext}")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        model_path = 'best.pt'
        model = YOLO(model_path)

        summary = []
        speed = ""

        if input_type == "Image":
            image, summary = process_image(model, temp_path)
            result_path = os.path.join(tempfile.gettempdir(), "result.jpg")
            cv2.imwrite(result_path, image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(temp_path, caption="Original Image", width=320)
            with col2:
                st.image(image, caption="Detected Image", width=320)

            with open(result_path, "rb") as img_file:
                download_placeholder.download_button(label="📥 Download Detection Output",
                                                     data=img_file,
                                                     file_name="detection_result.jpg",
                                                     mime="image/jpeg",
                                                     use_container_width=True)
        else:
            cap = cv2.VideoCapture(temp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps else 1
            speed = f"{round(10 + duration * 0.5, 2)} knots"  # Fake speed based on duration
            cap.release()
            st.video(temp_path)

        object_counts = {}
        avg_conf = 0
        largest_length = 0
        has_tonnage = False

        for item in summary:
            label = item['Object']
            object_counts[label] = object_counts.get(label, 0) + 1
            avg_conf += item['Confidence (%)']
            if label in ['ship', 'cargo']:
                has_tonnage = True
                largest_length = max(largest_length, item['Length (m)'])

        total_objects = sum(object_counts.values())
        ship_count = object_counts.get('ship', 0)
        cargo_count = object_counts.get('cargo', 0)
        clouds = object_counts.get('cloud', 0)
        islands = object_counts.get('island', 0)
        avg_conf = round(avg_conf / total_objects, 2) if total_objects else 0

        detected_objects_list = [f"**{obj.capitalize()}**: {count}" for obj, count in object_counts.items()]
        detected_objects_str = "  \n".join(detected_objects_list)

        # Add length and coordinates to the detection summary
        object_details = [f"**{item['Object'].capitalize()}:** Length: {item['Length (m)']} m, Coordinates: ({item['Latitude']}, {item['Longitude']})" for item in summary]
        object_details_str = "  \n".join(object_details)

        weather_status = "Clear Skies ☀️" if clouds < 2 else "Cloudy ☁️"
        risk_alert = ""
        if has_tonnage:
            risk_alert = "No Collision Risks 🚫" if ship_count + cargo_count < 3 else "Caution: High Density ⚠️"

        recent_time = datetime.utcnow().strftime('%Y-%m-%d')

        # Unified and cleaned up Detection Summary
        st.subheader("📊 Detection Summary")
        st.markdown(f"""
        **Detection Overview:**
        - **Total Objects Detected:** {total_objects}  
        - **Avg Confidence:** {avg_conf}%  
        - **Max Ship Length:** {largest_length} m  
        - **Weather:** {weather_status}  
        - **Risk:** {risk_alert}  
        - **Detected At:** {recent_time} UTC  
        - **Speed:** {speed}  (For Video)
        """)

        # Collapsible section for detailed detection results
        with st.expander("View Detailed Object Information"):
            for item in summary:
                st.write(f"**{item['Object'].capitalize()}**: Length: {item['Length (m)']} m, Coordinates: ({item['Latitude']}, {item['Longitude']})")

else:
    st.info("Please upload an image or video to start detection.")
