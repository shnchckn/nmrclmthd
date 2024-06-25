import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# Define tone ranges
tone_ranges = [
    (0, 29),    # Blacks
    (30, 90),   # Shadows
    (91, 164),  # Midtones
    (165, 225), # Highlights
    (226, 255)  # Whites
    ]

# Split the image by tone ranges
def tone_segment(image, tone_ranges):
    segmented_images = []
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (min, max) in tone_ranges:
        mask = cv2.inRange(grayscale, min, max)
        segmented_images.append(cv2.bitwise_and(image, image, mask=mask))
    return segmented_images

# Get the dominant color for each tone range
def get_dominant_color(image):
    pixels = image.reshape(-1, 3)
    pixels = pixels[pixels.sum(axis=1) > 0]  # Remove black pixels
    if len(pixels) == 0:   # Return (black) as none if no pixels
        return [0, 0, 0]  
    
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_.astype(int)
    
    return dominant_color

# Detect keypoints using SIFT
def get_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    return keypoints

# Filter keypoints by regions of each dominant color 
def filter_keypoints(image, keypoints, dominant_colors):
    filtered_keypoints = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        pixel_color = image[y, x]
        for color in dominant_colors:
            if np.all(np.abs(pixel_color - color) <= 30):   #Threshold is defined through trial and error
                filtered_keypoints.append(kp)
                break
    return filtered_keypoints

# Highlight dominant colors in the image
def highlight_regions(image, dominant_colors):
    mask = np.zeros_like(image[:, :, 0])
    for color in dominant_colors:
        mask |= cv2.inRange(image, color - 30, color + 30)
    return cv2.bitwise_and(image, image, mask=mask)

# Display the dominant colors by plotting
def plot_colors(dominant_colors, labels):
    fig, ax = plt.subplots(1, len(dominant_colors), figsize=(15, 3))
    for i, color in enumerate(dominant_colors):
        color_patch = np.zeros((100, 100, 3), dtype=int)
        color_patch[:, :] = color
        ax[i].imshow(color_patch)
        ax[i].axis('off')
        ax[i].set_title(labels[i])
    st.pyplot(fig)
    
# Display the rgb values of dominant colors
def rgb_values(dominant_colors, tone_labels):
    return{tone_labels[i]: {
            "r": int(dominant_colors[i][0][0]),
            "g": int(dominant_colors[i][0][1]),
            "b": int(dominant_colors[i][0][2])
        } for i in range(len(tone_labels))
    }

# GUI
st.title("Dominant Color and Keypoints Detection")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    segmented_images = tone_segment(image_rgb, tone_ranges)
    dominant_colors = [get_dominant_color(segmented_image) for segmented_image in segmented_images]
    tone_labels = ['Blacks', 'Shadows', 'Midtones', 'Highlights', 'Whites']
    plot_colors(dominant_colors, tone_labels)

    keypoints = get_keypoints(image_rgb)
    filtered_keypoints = filter_keypoints(image_rgb, keypoints, dominant_colors)
    
    highlighted_image = highlight_regions(image_rgb, dominant_colors)
    image_with_keypoints = cv2.drawKeypoints(highlighted_image, filtered_keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    st.image([image_rgb, highlighted_image, image_with_keypoints], caption=["Original Image", "Dominant Colors", "Keypoints"], use_column_width=True)
    
    rgb_values = rgb_values(dominant_colors, tone_labels)
    st.write("Dominant Colors (RGB) by Tone Ranges:", rgb_values)