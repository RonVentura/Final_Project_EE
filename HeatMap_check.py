import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import time
import tracemalloc
from PIL import Image
import os
import pandas as pd

def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0), image.size  # Return original size as well

def get_gradcam_heatmap(model, image, target_layer):
    gradients = None
    features = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    outputs = model(image)
    pred_class = outputs.argmax(dim=1)
    one_hot_output = torch.zeros(outputs.size(), dtype=torch.float)
    one_hot_output[0][pred_class] = 1
    model.zero_grad()
    outputs.backward(gradient=one_hot_output, retain_graph=True)

    handle_forward.remove()
    handle_backward.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(features.size(1)):
        features[:, i, :, :] *= pooled_gradients[i]

    heatmap = features.detach().cpu().numpy().mean(axis=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def display_heatmap(heatmap, image_path, output_path, alpha=0.4):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(output_path, superimposed_img)

def extract_high_activation_areas(heatmap, original_size, threshold=0.7):
    high_activation = heatmap > threshold
    nonzero_coords = np.argwhere(high_activation)
    top_left = np.min(nonzero_coords, axis=0)
    bottom_right = np.max(nonzero_coords, axis=0)

    heatmap_height, heatmap_width = heatmap.shape
    orig_width, orig_height = original_size

    top_left_x = int((top_left[1] / heatmap_width) * orig_width)
    top_left_y = int((top_left[0] / heatmap_height) * orig_height)
    bottom_right_x = int((bottom_right[1] / heatmap_width) * orig_width)
    bottom_right_y = int((bottom_right[0] / heatmap_height) * orig_height)

    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

def inference_on_sampling_folders(model, test_data_dir):
    sampling_percent_folders = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])
    results = []
    for percent_folder in sampling_percent_folders:
        if percent_folder == "heatmaps":
            continue
        sampling_ratio = int(percent_folder.split('_')[1])
        sampling_folder_path = os.path.join(test_data_dir, percent_folder)
        class_folders = sorted([d for d in os.listdir(sampling_folder_path) if os.path.isdir(os.path.join(sampling_folder_path, d))])
        if not class_folders:
            print("No class folders found in:", sampling_folder_path)
            continue
        for class_folder in class_folders:
            class_folder_path = os.path.join(sampling_folder_path, class_folder)
            image_files = [f for f in os.listdir(class_folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not image_files:
                print("No image files found in class folder:", class_folder_path)
                continue

            for image_file in image_files:
                image_path = os.path.join(class_folder_path, image_file)
                image, original_size = preprocess_image(image_path)
                start_time = time.time()
                tracemalloc.start()
                outputs = model(image)
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                inference_time = end_time - start_time
                memory_usage = peak / 10**6  # Convert bytes to megabytes
                pred_class = outputs.argmax(dim=1).item()
                accuracy = 1 if pred_class == class_folders.index(class_folder) else 0

                heatmap = get_gradcam_heatmap(model, image, model.layer4[2].conv3)  # Target the last conv layer in the bottleneck
                heatmap_output_dir = os.path.join('heatmaps', percent_folder, class_folder)
                os.makedirs(heatmap_output_dir, exist_ok=True)
                output_path = os.path.join(heatmap_output_dir, f'{image_file}_heatmap.jpg')
                display_heatmap(heatmap, image_path, output_path)
                top_left, bottom_right = extract_high_activation_areas(heatmap, original_size)

                results.append({
                    'Image Path': image_path,
                    'Top Left (x, y)': top_left,
                    'Bottom Right (x, y)': bottom_right,
                    'Inference Time (s)': inference_time,
                    'Memory Usage (MB)': memory_usage,
                    'Accuracy': accuracy,
                    'Heatmap Path': output_path
                })
    return results

if __name__ == '__main__':
    model_path = 'C://Users//Ronve//PycharmProjects//THE_Project//TrainOnSampled//sampling_60//pythorch_model_60_per.pth'
    test_data_dir = 'C://Users//Ronve//PycharmProjects//THE_Project//data-split//test'
    heatmaps_directory = os.path.join(test_data_dir, 'heatmaps')
    os.makedirs(heatmaps_directory, exist_ok=True)
    model = load_model(model_path, 4)
    results = inference_on_sampling_folders(model, test_data_dir)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    excel_path = os.path.join(test_data_dir, 'heatmap_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f'Results saved to {excel_path}')
