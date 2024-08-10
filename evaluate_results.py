import os
import torch
import torch.nn as nn
import time
import tracemalloc
import pandas as pd
from PIL import Image
from torchvision import models, transforms

def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def inference_on_sampling_folders(model, test_data_dir):
    sampling_folders = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])
    results = []
    for sampling_folder in sampling_folders:
        sampling_folder_path = os.path.join(test_data_dir, sampling_folder)
        class_folders = sorted([d for d in os.listdir(sampling_folder_path) if os.path.isdir(os.path.join(sampling_folder_path, d))])
        print(f"Processing Folder: {sampling_folder}")
        if not class_folders:
            print("No class folders found in:", sampling_folder_path)
            continue
        inference_times = []
        memory_usages = []
        accuracies = []
        for class_folder in class_folders:
            class_folder_path = os.path.join(sampling_folder_path, class_folder)
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not image_files:
                print("No image files found in class folder:", class_folder_path)
                continue
            images = []
            labels = []
            for image_file in image_files:
                image_path = os.path.join(class_folder_path, image_file)
                image = Image.open(image_path).convert('RGB')  # Ensure RGB format
                image = transforms.Resize(256)(image)
                image = transforms.CenterCrop(224)(image)
                image = transforms.ToTensor()(image)
                image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
                images.append(image)
                labels.append(class_folders.index(class_folder))  # Convert class folder name to label index
            images = torch.stack(images)
            labels = torch.tensor(labels)
            start_time = time.time()
            tracemalloc.start()
            outputs = model(images)
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            inference_time = end_time - start_time
            print(inference_time)
            memory_usage = peak / 10**6  # Convert bytes to megabytes
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / len(labels)
            inference_times.append(inference_time)
            memory_usages.append(memory_usage)
            accuracies.append(accuracy)
        if inference_times and memory_usages and accuracies:
            avg_inference_time = sum(inference_times) / len(inference_times)
            avg_memory_usage = sum(memory_usages) / len(memory_usages)
            avg_accuracy = sum(accuracies) / len(accuracies)
            results.append({
                'Folder Name': sampling_folder,
                'Inference Time (s)': avg_inference_time,
                'Memory Usage (MB)': avg_memory_usage,
                'Accuracy (%)': avg_accuracy
            })
    return results

if __name__ == '__main__':
    model_path = 'C://Users//Ronve//PycharmProjects//THE_Project//PyTorch_train//resnet50_custom_dataset.pth'
    test_data_dir = 'C://Users//Ronve//PycharmProjects//THE_Project//TODO//new_diagonal_black'
    model = load_model(model_path, 4)
    results = inference_on_sampling_folders(model, test_data_dir)

    # Save results to Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(test_data_dir, 'sampling_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f'Results saved to {excel_path}')
