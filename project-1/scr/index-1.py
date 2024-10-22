# Import necessary libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('yolo11n.pt')  # YOLOv8n is the small version, you can choose a larger model if needed

# Load video or image (provide the path to the video file)
video_path = r'assets\input.mp4'  # Path to the video file
cap = cv2.VideoCapture(video_path)


# Get video properties for saving the output video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Define the codec and create VideoWriter object to save the output video
output_path = 'assets\output.mp4'  # Path to the output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Function to plot the detection results on the frame
def plot_detection(frame, results):
    for result in results:
        boxes = result.boxes  # Detected boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Coordinates of the box
            label = model.names[int(box.cls)]  # Object label
            conf = box.conf[0]  # Confidence score
            
            # Filter for cars (usually labeled as 'car')
            if label == 'car':
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

# Perform object detection on the video and save the output
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform prediction with the model
    results = model(frame)

    # Draw detected objects
    frame = plot_detection(frame, results)

    # Write the processed frame to the output video file
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")