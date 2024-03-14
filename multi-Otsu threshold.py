import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

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
#video_path = 'C:\\Users\\Lenovo Thinkpad T490\\OneDrive\\Bureau\\research_project\\output_video.mp4'
video_path = 'C:\\Users\\Lenovo Thinkpad T490\\OneDrive\\Bureau\\research_project\\vids_project\\Mouillage_QL_INOX_20230512_10_25000_0_RC_1_tilt_rotate.avi'
frame = extract_random_frame(video_path)

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply multi-Otsu thresholding
thresholds = threshold_multiotsu(gray, classes=3)

# Generate binary masks based on the thresholds
binary_masks = [(gray >= thresholds[i]) & (gray <= thresholds[i+1]) for i in range(len(thresholds)-1)]

# Combine binary masks into a single binary image
binary_image = np.zeros_like(gray, dtype=np.uint8)
for mask in binary_masks:
    binary_image |= mask.astype(np.uint8)

# Detecting the water drop
threshold_area = 60
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > threshold_area:
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)  # Blue color

# Detecting the surface
threshold1 = 100
threshold2 = 100
rho = 1
threshold = 10
theta = 1
minLineLength = 10
maxLineGap = 0

edges = cv2.Canny(binary_image, threshold1, threshold2)
lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color

# Display the annotated image
cv2.imshow('Annotated Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
