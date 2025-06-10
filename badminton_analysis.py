from ultralytics import YOLO
import cv2
from collections import defaultdict
import os
os.makedirs("output", exist_ok=True)

# Load the trained YOLOv5 model
model = YOLO('best.pt').to('cuda')  # path to your custom-trained weights

# Input and output video paths
input_path = 'input_video/raw.mp4'
output_path = 'output/result_video.mp4'

# Open the input video
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Dictionary to count classes
class_counts = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        class_counts[cls_name] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{cls_name} {conf:.2f}"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

# Save or print class counts
print("\n📊 Shot Summary:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")

# Optional: Save to file
with open("output/summary.txt", "w") as f:
    for cls, count in class_counts.items():
        f.write(f"{cls}: {count}\n")

print("\n✅ Done! Annotated video and summary saved in 'output/' folder.")
