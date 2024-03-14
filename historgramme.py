import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to extract a random frame from a video
def extract_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_index = np.random.randint(0, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame

# Read the video and extract a random frame
video_path = 'C:\\Users\\Lenovo Thinkpad T490\\OneDrive\\Bureau\\research_project\\output_video.mp4'
frame = extract_random_frame(video_path)

# Convert the frame to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Compute histogram
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plot histogram
plt.plot(histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Pixel Intensities')
plt.show()
