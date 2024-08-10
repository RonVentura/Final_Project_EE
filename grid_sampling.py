import os
import random
from PIL import Image

# Define the path to the images folder
images_folder = "C://Users//Ronve//PycharmProjects//THE_Project//test_100_percent"

# Define the arrays
sampling_percentages = [20, 30, 40, 50, 60, 70, 80]

# Function to create a new folder
def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to divide the image into squares and calculate mean value for each square
def calculate_square_means(image, square_size=30):
    width, height = image.size
    square_means = {}

    for x in range(0, width, square_size):
        for y in range(0, height, square_size):
            left = x
            upper = y
            right = min(x + square_size, width)
            lower = min(y + square_size, height)

            square = image.crop((left, upper, right, lower))
            mean_value = tuple(int(sum(c) / len(c)) for c in zip(*square.getdata()))

            square_means[(left, upper, right, lower)] = mean_value

    return square_means

# Function to sample pixels in a grid pattern
def sample_pixels_in_grid(image, sampling_percentage, square_means, block_size=30):
    width, height = image.size
    sampled_image = Image.new('RGB', (width, height), (0, 0, 0))

    for x in range(0, width, block_size):
        for y in range(0, height, block_size):
            if random.random() < sampling_percentage / 100:
                block = image.crop((x, y, min(x + block_size, width), min(y + block_size, height)))
                sampled_image.paste(block, (x, y))
            else:
                for square, mean_value in square_means.items():
                    if x >= square[0] and x < square[2] and y >= square[1] and y < square[3]:
                        for i in range(x, min(x + block_size, width)):
                            for j in range(y, min(y + block_size, height)):
                                sampled_image.putpixel((i, j), mean_value)

    return sampled_image

# Function to process images
def process_images():
    for percentage in sampling_percentages:
        new_folder = f"grid_sampling_{percentage}_percent"
        create_new_folder(new_folder)

        for root, dirs, files in os.walk(images_folder):
            for name in files:
                image_path = os.path.join(root, name)
                relative_path = os.path.relpath(image_path, images_folder)

                output_folder = os.path.join(new_folder, os.path.dirname(relative_path))
                create_new_folder(output_folder)

                output_image_path = os.path.join(output_folder, name)
                image = Image.open(image_path)
                square_means = calculate_square_means(image)
                sampled_image = sample_pixels_in_grid(image, percentage, square_means)
                sampled_image.save(output_image_path)

# Run the process_images function
process_images()
