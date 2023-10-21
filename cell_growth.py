#Testing georges branch
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

image_files = glob.glob('/Users/gmacheta/Desktop/untitled folder/SHSY5Y Rep 1/*.tif')
results = []

# Extract image numbers from filenames
image_numbers = [int(image_file.split('_')[1].split('.')[0]) for image_file in image_files]

# Sort the image files based on the image number
sorted_image_files = [x for _, x in sorted(zip(image_numbers, image_files))]

for image_file in sorted_image_files:
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to the grayscale image
    threshold_value = 130  # Adjust this value
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Calculate the number of black pixels (background) in the thresholded image
    num_black_pixels = np.sum(thresh == 0)

    # Calculate the total number of pixels in the image
    total_pixels = thresh.size

    # Calculate the number of cells as the difference between total pixels and black pixels
    num_cells = total_pixels - num_black_pixels

    results.append(num_cells)

# Sort the results and image_numbers based on the number of cells (from least to most)

plt.scatter(range(1, len(sorted_image_files) + 1), results, marker='o', s=30, c='b', label='Data Points')
plt.xlabel('Image Order')
plt.ylabel('Number of Cells')
plt.title('Cell Growth')
plt.grid(True)
plt.legend()
plt.show()
