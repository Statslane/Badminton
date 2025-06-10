import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path

st.set_page_config(page_title="Badminton Shot Detector", layout="wide")
st.title("🏸 Badminton Shot Detection using YOLOv8")

# Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.video(uploaded_file)
    # Temporary input/output files
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input_path = temp_input.name
    temp_output_path = temp_input_path.replace(".mp4", "_processed.mp4")



    # Load YOLOv8 model
    model = YOLO('best.pt').to('cuda')

    # Open input video
    cap = cv2.VideoCapture(temp_input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Shot count dictionary
    class_counts = defaultdict(int)

    progress = st.progress(0, text="🔍 Processing video...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            class_counts[cls_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{cls_name} {conf:.2f}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

        current_frame += 1
        progress.progress(min(current_frame / total_frames, 1.0), text="🔍 Processing video...")

    cap.release()
    out.release()
    progress.empty()

    # Display summary in grid
    st.subheader("📊 Shot Summary")
    cols = st.columns(3)
    for i, (cls, count) in enumerate(class_counts.items()):
        with cols[i % 3]:
            st.metric(label=cls, value=int(count/3))

    # Download button
    with open(temp_output_path, 'rb') as f:
        st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4")
