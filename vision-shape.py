import cv2
import numpy as np

# Function to track and adjust camera position based on the detected ball
def track_and_adjust(frame, center, window_width, window_height):
    # Calculate the new position of the frame based on the ball's center
    new_x = max(0, min(frame.shape[1] - window_width, center[0] - window_width // 2))
    new_y = max(0, min(frame.shape[0] - window_height, center[1] - window_height // 2))
    
    # Adjust the frame position
    adjusted_frame = frame[new_y:new_y + window_height, new_x:new_x + window_width].copy()

    return adjusted_frame

# Function to track the ball using Hough Circle Transform
def track_ball(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10, maxRadius=100)

    # Initialize center of the ball as None
    center = None

    # If circles are detected
    if circles is not None:
        # Convert coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")

        # Loop over all detected circles
        for (x, y, r) in circles:
            # Update the center coordinates
            center = (x, y)

    return frame, center

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define window width and height for tracking
window_width = 400
window_height = 400

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Check if frame is received properly
    if not ret:
        break

    # Track the ball in the frame
    tracked_frame, center = track_ball(frame)
    
    # Adjust camera position based on object position
    if center is not None:
        frame_to_show = track_and_adjust(frame, center, window_width, window_height)
    else:
        frame_to_show = frame

    # Display the resulting frame
    cv2.imshow('Ball Tracking', frame_to_show)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
