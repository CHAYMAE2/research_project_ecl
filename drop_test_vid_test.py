import cv2

# Read the video file
video_path = r'C:\Users\Lenovo Thinkpad T490\OneDrive\Bureau\research_project\output_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Initialize the video writer for saving the annotated video
output_video_path = 'annotated_video.mp4'
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while True:
    # Read the next frame from the video
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        break

    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 9)

    # Preprocessing (median blur)
    binary_image = cv2.medianBlur(binary_image, 5)  # Apply median blur to remove noise

    threshold_area = 300 # Adjust as needed

    # Detecting the water drop
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold_area:
            # Draw filled contour
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), -1)  # Fill contour with blue color

    # Write the annotated frame to the output video file
    video_writer.write(frame)

    # Display the annotated frame
    cv2.imshow('Annotated Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
