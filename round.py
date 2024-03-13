import cv2
import numpy as np

# Function to track the ball and adjust camera position
def track_and_adjust(frame, center, frame_width, frame_height, window_width, window_height):
    # Calculate the center of the frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
   
    # Calculate the offset of the object from the frame center
    offset_x = center[0] - frame_center_x
    offset_y = center[1] - frame_center_y
   
    # Calculate the new position of the frame
    new_x = max(0, min(frame_width - window_width, offset_x))
    new_y = max(0, min(frame_height - window_height, offset_y))
   
    # Adjust the frame position
    adjusted_frame = frame[new_y:new_y+window_height, new_x:new_x+window_width]
   
    return adjusted_frame

# Function to track the ball
def track_ball(frame, color_lower, color_upper):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only the specified color range
    mask = cv2.inRange(hsv, color_lower, color_upper)

    # Morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize center of the ball as None
    center = None

    # If contours are found
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Calculate moments of the largest contour
        M = cv2.moments(c)

        # Calculate center of the ball
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Draw the circle and centroid on the frame
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return frame, center

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define window width and height for tracking
window_width = 400
window_height = 400

# Define the color range for tennis ball and basketball
tennis_ball_lower = np.array([30, 50, 50])
tennis_ball_upper = np.array([45, 255, 255])

basketball_lower = np.array([0, 0, 0])
basketball_upper = np.array([255, 255, 255])

# Set default color range to tennis ball
lower_color = tennis_ball_lower
upper_color = tennis_ball_upper

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Check if frame is received properly
    if not ret:
        break

    # Track the ball in the frame
    tracked_frame, center = track_ball(frame, lower_color, upper_color)
   
    # Adjust camera position based on object position
    if center is not None:
        frame_to_show = track_and_adjust(frame, center, frame_width, frame_height, window_width, window_height)
    else:
        frame_to_show = frame

    # Display the resulting frame
    cv2.imshow('Ball Tracking', frame_to_show)

    # Calibration: Press 'c' to switch between tennis ball and basketball tracking
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if lower_color == tennis_ball_lower:
            lower_color = basketball_lower
            upper_color = basketball_upper
        else:
            lower_color = tennis_ball_lower
            upper_color = tennis_ball_upper
    # Exit if 'q' is pressed
    elif key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
