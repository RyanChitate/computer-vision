import cv2
import numpy as np

# Function to track the ball and adjust camera position


def track_and_adjust(frame, center, prev_center, frame_width, frame_height, window_width, window_height):
    # If the center is None or previous center is None, keep the camera centered
    if center is None or prev_center is None:
        new_x = (frame_width - window_width) // 2
        new_y = (frame_height - window_height) // 2
    else:
        # Calculate the new position of the frame based on the object's movement
        offset_x = center[0] - prev_center[0]
        offset_y = center[1] - prev_center[1]
        new_x = max(0, min(frame_width - window_width,
                    prev_center[0] - offset_x))
        new_y = max(0, min(frame_height - window_height,
                    prev_center[1] - offset_y))

    # Adjust the frame position
    adjusted_frame = frame[new_y:new_y+window_height, new_x:new_x+window_width]

    return adjusted_frame

# Function to track the ball


def track_ball(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of grey color in HSV
    lower_color = np.array([0, 0, 50])
    upper_color = np.array([179, 50, 220])

    # Threshold the HSV image to get only grey color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

# Initialize previous center
prev_center = None

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Check if frame is received properly
    if not ret:
        break

    # Track the ball in the frame
    tracked_frame, center = track_ball(frame)

    # Adjust camera position based on object position
    frame_to_show = track_and_adjust(
        frame, center, prev_center, frame_width, frame_height, window_width, window_height)

    # Update previous center
    prev_center = center

    # Display the resulting frame
    cv2.imshow('Object Tracking', frame_to_show)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
