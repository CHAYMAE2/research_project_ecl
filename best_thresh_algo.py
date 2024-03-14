import cv2
from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
import numpy as np 

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

# Apply the try_all_threshold function
fig, ax = try_all_threshold(gray_image, figsize=(10, 8), verbose=False)
plt.show()
