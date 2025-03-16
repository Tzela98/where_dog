import cv2
import torch
import numpy as np
import serial
import atexit
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/exp7/weights/best.pt')

# Initialize serial communication via USB
try:
    ser = serial.Serial('/dev/cu.usbserial-0001', 115200)
    print("Serial connection established.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

# Open camera
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Smoothing variables
SMOOTHING_WINDOW_SIZE = 5  # Number of frames to average over
step_history = []  # Store recent steps for smoothing

# Function to track the dog and return its horizontal center
def track_dog(frame, model):
    # Run YOLOv5 inference
    results = model(frame)

    # Extract bounding boxes, labels, and confidence scores
    detections = results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, confidence, class]

    dog_class_id = 0  # Replace with the correct class ID for "dog"
    dog_detections = [det for det in detections if det[5] == dog_class_id]

    if len(dog_detections) > 0:
        # Get the first detected dog (you can add logic to handle multiple dogs)
        x1, y1, x2, y2, confidence, _ = dog_detections[0]

        # Calculate the center of the bounding box (horizontal position)
        dog_center_x = (x1 + x2) / 2

        return results, dog_center_x
    else:
        return None, None  # No dog detected

# Function to calculate the required stepper motor movement
def calculate_stepper_movement(dog_center_x, frame_width):
    frame_center_x = frame_width / 2
    offset = dog_center_x - frame_center_x

    # Define a scaling factor to convert pixel offset to stepper steps
    scaling_factor = 0.1
    steps = int(offset * scaling_factor)

    return steps

# Function to smooth steps using a moving average
def smooth_steps(new_steps):
    global step_history

    # Add new steps to the history
    step_history.append(new_steps)

    # Keep only the last SMOOTHING_WINDOW_SIZE steps
    if len(step_history) > SMOOTHING_WINDOW_SIZE:
        step_history.pop(0)

    # Calculate the average of the last SMOOTHING_WINDOW_SIZE steps
    smoothed_steps = int(sum(step_history) / len(step_history))

    return smoothed_steps

# Cleanup function to close serial port on exit
def cleanup():
    if ser.is_open:
        ser.write("0\n".encode())  # Send 0 steps to stop the motor
        ser.close()
        print("Serial port closed.")

# Register cleanup function
atexit.register(cleanup)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track the dog
    track_result, dog_center_x = track_dog(frame, model)

    if track_result is not None:
        # Render the results (for visualization)
        frame_with_result = track_result.render()[0]

        # Create a writable copy of the rendered frame
        frame_with_result = frame_with_result.copy()
    else:
        frame_with_result = frame  # Use the original frame if no dog is detected

    if dog_center_x is not None:
        # Calculate stepper motor steps
        steps = calculate_stepper_movement(dog_center_x, frame_width)

        # Smooth the steps
        smoothed_steps = smooth_steps(steps)

        # Send smoothed steps to ESP32
        try:
            ser.write(f"{smoothed_steps}\n".encode())
            print(f"Sent smoothed steps: {smoothed_steps}")
        except Exception as e:
            print(f"Error sending data to ESP32: {e}")
    else:
        # Send 0 steps when no dog is detected
        try:
            ser.write("0\n".encode())
            print("Sent steps: 0 (no dog detected)")
        except Exception as e:
            print(f"Error sending data to ESP32: {e}")

    # Draw the bounding box and center line (for visualization)
    if dog_center_x is not None:
        cv2.rectangle(frame_with_result, (int(dog_center_x - 50), frame_height // 2 - 50),
                      (int(dog_center_x + 50), frame_height // 2 + 50), (0, 255, 0), 2)
        cv2.line(frame_with_result, (int(dog_center_x), 0), (int(dog_center_x), frame_height), (0, 255, 0), 2)

    # Draw the frame center line (for visualization)
    cv2.line(frame_with_result, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Dog Tracking", frame_with_result)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()