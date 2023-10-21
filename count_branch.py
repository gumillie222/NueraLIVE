from skimage.filters import frangi, hessian, sato, meijering
from skimage import io
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def count_branches(image):
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dns_greyscale = cv2.fastNlMeansDenoising(greyscale)
    filtered_image  = meijering(dns_greyscale, [1], black_ridges = True)
    return np.count_nonzero((filtered_image > 2/255) & (filtered_image < 0.05))

# def process_image(image_file):
#     image = cv2.imread(image_file, cv2.IMREAD_COLOR)
#     return count_branches(image)

def extract_image_numbers(image_file):
    image_number = int(''.join(filter(lambda x: x in '0123456789', image_file.split('_')[5])))
    image_culture_number = int(''.join(filter(lambda x: x in '0123456789', image_file.split('_')[6])))
    return image_number, image_culture_number, image_file

def generate_plot_branch():
    # Process the images
    # image_files = glob.glob('./SHSY5Y_Rep1/*.tif')
    image_files = glob.glob('./SHSY5Y_Rep1/*.tif')
    results = [[],[],[],[]]

    with ThreadPoolExecutor(max_workers=4) as executor:
        data = list(executor.map(extract_image_numbers, image_files))

    sorted_data = sorted(data, key=lambda x: x[0])

    for image_number, image_culture_number, image_file in sorted_data:
        if image_culture_number in [1, 2, 3, 4]:
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            num_branches = count_branches(image)
            results[image_culture_number-1].append(num_branches)

    for i in range(4):
        plt.scatter(range(1, len(results[i]) + 1), results[i], marker='o', s=30, label=f'Culture {i + 1}')


    # image_numbers = [int(''.join(filter(lambda x: x in '0123456789', image_file.split('_')[5]))) for image_file in image_files]
    # image_culture_numbers = [int(''.join(filter(lambda x: x in '0123456789', image_file.split('_')[6]))) for image_file in image_files]

    # # Sort the image files based on the image number
    # sorted_image_files = [x for _, x in sorted(zip(image_numbers, zip(image_files, image_culture_numbers)))]
    # print(sorted_image_files)

    # for image_file, image_culture_number in sorted_image_files:
    #     image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    #     num_branches = count_branches(image)
    #     if (image_culture_number in [1, 2, 3, 4]):
    #         results[image_culture_number-1].append(num_branches)


    # plt.scatter(range(1, len(results[0]) + 1), results[0], marker='o', s=30, c='b', label='Culture 1')
    # print('fnished 1')
    # plt.scatter(range(1, len(results[1]) + 1), results[1], marker='o', s=30, c='r', label='Culture 2')
    # plt.scatter(range(1, len(results[2]) + 1), results[2], marker='o', s=30, c='g', label='Culture 3')
    # plt.scatter(range(1, len(results[3]) + 1), results[3], marker='o', s=30, c='c', label='Culture 4')
    # 
    plt.xlabel('Image Order')
    plt.ylabel('Number of Pixels of Neurites')
    plt.title('Cell Branches')
    plt.grid(True)
    plt.legend()
    plt.show()

    graph_filename = 'static/branch_count.png'
    plt.savefig(graph_filename)
    plt.close()


def main():
    generate_plot_branch()

#if __name__ == "__main__":
main()