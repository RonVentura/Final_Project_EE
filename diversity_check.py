import cv2
import numpy as np
import os
import pandas as pd

def compute_patch_diversity(patch):

    # Calculate histogram for non-black pixels (excluding pixels with value 0)
    hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Compute entropy as a measure of diversity
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    return entropy

def find_most_diverse_patch(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Determine patch dimensions (assuming 3x4 grid for simplicity)
    height, width, _ = img.shape
    num_rows, num_cols = 3, 4
    patch_height = height // num_rows
    patch_width = width // num_cols

    max_diversity = -1
    most_diverse_patch = None
    top_left = None
    bottom_right = None

    # Iterate through each patch
    for r in range(num_rows):
        for c in range(num_cols):
            # Extract patch from the image
            patch = img[r * patch_height:(r + 1) * patch_height, c * patch_width:(c + 1) * patch_width]

            # Compute diversity (using entropy as an example)
            diversity = compute_patch_diversity(patch)

            # Check if this patch has higher diversity
            if diversity > max_diversity:
                max_diversity = diversity
                most_diverse_patch = patch
                # Calculate top-left and bottom-right coordinates
                top_left = (c * patch_width, r * patch_height)
                bottom_right = ((c + 1) * patch_width, (r + 1) * patch_height)

    return most_diverse_patch, top_left, bottom_right


def process_images_in_folder(data_dir):
    results = []
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            if dir.startswith("sampling_"):
                percent_folder = os.path.join(root, dir)
                for class_folder in os.listdir(percent_folder):
                    class_folder_path = os.path.join(percent_folder, class_folder)
                    for file in os.listdir(class_folder_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            image_path = os.path.join(class_folder_path, file)
                            most_diverse_patch, top_left, bottom_right = find_most_diverse_patch(image_path)
                            if most_diverse_patch is not None:
                                results.append({
                                    'Image Path': image_path,
                                    'Most Diverse Patch': most_diverse_patch,
                                    'Top Left (x, y)': top_left,
                                    'Bottom Right (x, y)': bottom_right
                                })
    return results


if __name__ == '__main__':
    test_data_dir = 'C://Users//Ronve//PycharmProjects//THE_Project//samplingFillMean//samplingFillMean_forLowRate'
    results = process_images_in_folder(test_data_dir)

    # Optionally, save the most diverse patches
    for result in results:
        image_name = os.path.splitext(os.path.basename(result['Image Path']))[0]
        save_path = os.path.join(test_data_dir, f'{image_name}_most_diverse_patch.jpg')
        cv2.imwrite(save_path, result['Most Diverse Patch'])

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    excel_path = os.path.join(test_data_dir, 'most_diverse_patch_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f'Results saved to {excel_path}')