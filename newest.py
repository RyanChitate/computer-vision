import cv2
import numpy as np
import threading

# Function to track the ball and adjust camera position
def track_and_adjust(frame, center, prev_center, frame_width, frame_height, window_width, window_height):
    # Calculate the center of the frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
   
    # Calculate the offset of the object from the frame center
    if center is not None:
        offset_x = center[0] - frame_center_x
        offset_y = center[1] - frame_center_y
    else:
        # If the object is not found, maintain the previous center
        offset_x = prev_center[0] - frame_center_x
        offset_y = prev_center[1] - frame_center_y
    
    # Calculate the new position of the frame
    new_x = max(0, min(frame_width - window_width, offset_x))
    new_y = max(0, min(frame_height - window_height, offset_y))
   
    # Adjust the frame position
    adjusted_frame = frame[new_y:new_y+window_height, new_x:new_x+window_width]
   
    return adjusted_frame

# Function to track the ball
def track_ball(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    # Initialize center of the ball as None
    center = None

    # If circles are found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Take the first circle found
        (x, y, radius) = circles[0]

        # Calculate center of the ball
        center = (x, y)

        # Draw the circle on the frame
        cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return frame, center

# Function to continuously capture frames from the webcam
def capture_frames():
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define window width and height for tracking
window_width = 400
window_height = 400

# Initialize previous center as the center of the frame
prev_center = (frame_width // 2, frame_height // 2)

# Global variable to store the current frame
frame = None

# Create a thread for frame capturing
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

while True:
    # Wait for the frame to be captured
    if frame is None:
        continue

    # Track the ball in the frame
    tracked_frame, center = track_ball(frame)
   
    # Adjust camera position based on object position
    frame_to_show = track_and_adjust(frame, center, prev_center, frame_width, frame_height, window_width, window_height)
    
    # Update the previous center
    if center is not None:
        prev_center = center

    # Display the resulting frame
    cv2.imshow('Ball Tracking', frame_to_show)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
