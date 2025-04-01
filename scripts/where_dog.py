import cv2
import torch
import numpy as np
import serial
import atexit
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Configuration
SMOOTHING_WINDOW_SIZE = 5
SWEEP_THRESHOLD = 50
SWEEP_RANGE_START = 0
SWEEP_RANGE_END = 180
SWEEP_STEP_SIZE = 1

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/exp7/weights/best.pt')
if model is None:
    print("ERROR: Model failed to load")
    exit()

# Initialize serial communication via USB
try:
    ser = serial.Serial('/dev/cu.usbserial-0001', 115200, timeout=1)
    print("Serial connection established.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('Error: Camera failed to initilize')
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tracking variables
step_history_x = [0]*SMOOTHING_WINDOW_SIZE
step_history_y = [0]*SMOOTHING_WINDOW_SIZE
no_dog_counter = 0
current_position = SWEEP_RANGE_START
sweep_direction = 1 # 1 for right, -1 for left
sweep_flag = False

# Function to track the dog and return its horizontal center
def track_dog(frame, model):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, confidence, class]
    dog_class_id = 0  
    dog_detections = [det for det in detections if det[5] == dog_class_id]

    if len(dog_detections) > 0:
        # Get the first detected dog
        x1, y1, x2, y2, confidence, _ = dog_detections[0]

        # Calculate the center of the bounding box
        dog_center_x = (x1 + x2) / 2
        dog_center_y = (y1 + y2) / 2

        return results, dog_center_x, dog_center_y
    else:
        return None, None, None  # No dog detected

# Function to calculate the required stepper motor movement
def calculate_stepper_movement(dog_center, frame_width):
    frame_center = frame_width / 2
    offset = dog_center - frame_center

    # Define a scaling factor to convert pixel offset to stepper steps
    scaling_factor = 0.1
    steps = int(offset * scaling_factor)

    return steps

# Map sweep position (0-180) to steps (0-200)
def map_sweep_to_steps(position, sweep_range_start, sweep_range_end, step_range_start, step_range_end):
    return int(np.interp(position, [sweep_range_start, sweep_range_end], [step_range_start, step_range_end]))

# Function to smooth steps using a moving average
def smooth_steps(new_steps, step_history):
    # Add new steps to the history
    step_history.append(new_steps)

    # Keep only the last SMOOTHING_WINDOW_SIZE steps
    if len(step_history) > SMOOTHING_WINDOW_SIZE:
        step_history.pop(0)

    # Calculate the average of the last SMOOTHING_WINDOW_SIZE steps
    smoothed_steps = int(sum(step_history) / len(step_history))

    return smoothed_steps, step_history

# Cleanup function to close serial port on exit
def cleanup():
    if ser.is_open:
        ser.write("xy:0,0\n".encode())
        ser.close()
        print("Serial port closed.")
atexit.register(cleanup)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track the dog
    track_result, dog_center_x, dog_center_y = track_dog(frame, model)
    frame_display = track_result.render()[0].copy() if track_result else frame.copy()

    if dog_center_x and dog_center_y is not None:
        # Calculate stepper motor steps
        steps_x = calculate_stepper_movement(dog_center_x, frame_width)
        steps_y = calculate_stepper_movement(dog_center_y, frame_height)

        # Smooth the steps
        smoothed_steps_x, step_history_x = smooth_steps(steps_x, step_history_x)
        smoothed_steps_y, step_history_y = smooth_steps(steps_y, step_history_y)

        # Send smoothed steps to ESP32
        try:
            ser.write(f"xy:{steps_x},{steps_y}\n".encode())
            print(f"Moving: x={steps_x}, y={steps_y}")
            no_dog_counter = 0
            sweep_flag = False
        except Exception as e:
            print(f"Error sending data to ESP32: {e}")

    else:
        no_dog_counter += 1
        try:
            ser.write(f"xy:0,0\n".encode())
            print("No dog - holding position")
        except Exception as e:
            print(f"Serial error: {e}")

        if no_dog_counter >= SWEEP_THRESHOLD and not sweep_flag:
            sweep_flag = True
            current_position = SWEEP_RANGE_START
            sweep_direction = 1
            print('Starting Camera Sweep...')
    
    # Handle sweep movement
    if sweep_flag:
        current_position += sweep_direction * SWEEP_STEP_SIZE
        target_steps = int(np.interp(current_position, [SWEEP_RANGE_START, SWEEP_RANGE_END], [0, 200]))
        
        try:
            ser.write(f"xy:{target_steps},0\n".encode())  # Only move X axis
            print(f"Sweeping: {target_steps}")
        except Exception as e:
            print(f"Sweep error: {e}")

        # Reverse direction at boundaries
        if current_position >= SWEEP_RANGE_END or current_position <= SWEEP_RANGE_START:
            sweep_direction *= -1


    # Draw the bounding box and center line (for visualization)
    if dog_center_x is not None:
        cv2.rectangle(frame_display, (int(dog_center_x - 50), frame_height // 2 - 50),
                      (int(dog_center_x + 50), frame_height // 2 + 50), (0, 255, 0), 2)
        cv2.line(frame_display, (int(dog_center_x), 0), (int(dog_center_x), frame_height), (0, 255, 0), 2)

    # Draw the frame center line (for visualization)
    cv2.line(frame_display, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Dog Tracking", frame_display)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()