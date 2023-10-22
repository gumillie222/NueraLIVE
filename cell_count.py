import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# Function to count cells based on their shape
def count_cells(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold_value = 130 # Adjust this value
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_count = 0
    for contour in contours:
        # Approximate the contour as a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is almost circular or a bulge (e.g., having a certain number of vertices)
        if len(approx) >= 7:
            cell_count += 1

    return cell_count

def generate_plot (imagespath,imagename):
    
    image_files = glob.glob(os.path.join(imagespath,'*.tif'))
    results = []
    
    # Extract image numbers from filenames
    image_numbers = [int(''.join(filter(lambda x: x in '0123456789', image_file.split('_')[7]))) for image_file in image_files]
    
    # Sort the image files based on the image number
    sorted_image_files = [x for _, x in sorted(zip(image_numbers, image_files))]
   
    for image_file in sorted_image_files:
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        
        num_cells = count_cells(image)

        results.append(num_cells)
   
    # Create a scatter plot
    plt.scatter(range(1, len(sorted_image_files) + 1), results, marker='o', s=30, c='b', label='Data Points')
    plt.xlabel('Time')
    plt.ylabel('Number of Cells)')
    plt.title('Cell Growth')
    plt.grid(True)
    plt.legend()

    graph_filename = 'static/' + imagename + '.png'
    plt.savefig(graph_filename)
    plt.close()

