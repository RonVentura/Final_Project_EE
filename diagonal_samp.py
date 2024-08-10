import os
from PIL import Image

# Define the path to the images folder
images_folder = "C://Users//Ronve//PycharmProjects//THE_Project//test_100_percent"

# Define the arrays
rows_thickness = [(1, 2), (2, 1), (3, 1), (1, 3), (4, 1), (1, 4), (1, 1)]
percentage = [33, 66, 75, 25, 80, 20, 50]

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

# Function to sample pixels diagonally
def sample_pixels_diagonally(image, thick, square_means):
    width, height = image.size
    sampled_image = Image.new('RGB', (width, height), (0, 0, 0))

    sample_thick, unsample_thick = thick
    max_diagonal = width + height - 1

    for d in range(max_diagonal):
        sample = (d // (sample_thick + unsample_thick)) % 2 == 0
        dy = d % (sample_thick + unsample_thick)

        if dy < sample_thick:
            for x in range(width):
                y = height - 1 - (d - x)
                if 0 <= y < height:
                    pixel = image.getpixel((x, y))
                    sampled_image.putpixel((x, y), pixel)
        else:
            for x in range(width):
                y = height - 1 - (d - x)
                if 0 <= y < height:
                    for square, mean_value in square_means.items():
                        if square[0] <= x < square[2] and square[1] <= y < square[3]:
                            sampled_image.putpixel((x, y), (0,0,0))

    return sampled_image

# Function to process images
def process_images():
    for i in range(len(rows_thickness)):
        new_folder = f"sampling_{percentage[i]}_percent"
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
                sampled_image = sample_pixels_diagonally(image, rows_thickness[i], square_means)
                sampled_image.save(output_image_path)

# Run the process_images function
process_images()
