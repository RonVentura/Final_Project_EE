import cv2
import numpy as np
import os
import pandas as pd


def extract_interesting_area(image_path, output_dir):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (9, 9), 0)

    # Apply Sobel edge detection in x and y directions
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)

    # Combine Sobel x and y to get edges
    edges = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(edges)

    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(edges, 240, 255, cv2.THRESH_BINARY)

    # Apply morphological operations (dilation and erosion)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Closing operation

    # Save edges image
    filename = os.path.basename(image_path)
    edges_filename = f"edges_{filename}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, edges_filename)
    cv2.imwrite(output_path, thresh)
    print(f"Edges image saved: {output_path}")

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y), (x + w, y + h)
    else:
        return None, None


def process_images_in_folder(data_dir, output_dir):
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
                            dir1 = os.path.join(output_dir,dir,class_folder)
                            top_left, bottom_right = extract_interesting_area(image_path, dir1)
                            if top_left is not None and bottom_right is not None:
                                results.append({
                                    'Image Path': image_path,
                                    'Top Left (x, y)': top_left,
                                    'Bottom Right (x, y)': bottom_right
                                })
    return results


if __name__ == '__main__':
    test_data_dir = 'C://Users//Ronve//PycharmProjects//THE_Project//samplingFillMean//samplingFillMean_forLowRate'
    output_dir = os.path.join(test_data_dir, 'edges_detected')
    os.makedirs(output_dir, exist_ok=True)

    results = process_images_in_folder(test_data_dir, output_dir)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    excel_path = os.path.join(test_data_dir, 'edge_detection_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f'Results saved to {excel_path}')
