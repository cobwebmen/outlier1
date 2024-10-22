import cv2
from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO('yolov11n.pt')

# Path to the video file
video_path = r'assets\input.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Path to the output video file
output_path = 'assets\output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def plot_detection(frame, results):
    for result in results:
        boxes = result.boxes  # Detected boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Coordinates of the box
            label = model.names[int(box.cls)]  # Object label
            conf = box.conf[0]  # Confidence score
            
            # Filter for cars (usually labeled as 'car')
            if label == 'car':
                # Ensure rectangle coordinates are within frame boundaries
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
                
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection on the frame
        results = model(frame)
        
        # Plot detections on the frame
        frame = plot_detection(frame, results)
        
        # Write the frame to the output video
        out.write(frame)
    
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")