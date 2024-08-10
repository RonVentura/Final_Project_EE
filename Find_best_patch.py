import os
import torch
import torchvision.utils
from torchvision import transforms as T
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import pandas as pd

def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to divide an image into overlapping patches with dynamic size
def get_patches(image, num_patches, overlap):
    patches = []
    img_width, img_height = image.size
    patch_width = img_width // 2
    patch_height = img_height // 2

    #stride_x = int(patch_width + (overlap*patch_width))
    #stride_y = int(patch_height + (overlap*patch_height))

    for y in range(0, img_height, patch_height):
        for x in range(0, img_width, patch_width):
            patch = image.crop((x, y, x + patch_width, y + patch_height))
            patches.append((patch, (x, y)))
            #patch.show()
            if len(patches) == num_patches:
                return patches

    return patches

# Function to classify patches and find the best one
def classify_patches(model, patches):
    best_patch = None
    best_score = -1
    best_coords = None

    patches_l = []
    coords_l = []
    for patch, coords in patches:
        #patch.show()
        patch_tensor = T.Resize(256)(patch)
        patch_tensor = T.CenterCrop(224)(patch_tensor)
        patch_tensor = T.ToTensor()(patch_tensor)
        patch_tensor = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(patch_tensor)
        patches_l.append(patch_tensor)
        coords_l.append(coords)
    patches_l = torch.stack(patches_l)
    outputs = model(patches_l)
    outputs, _ = torch.max(outputs, 1)
    s = 0
    for i in range(len(outputs)):
        if outputs[i] > s:
            s = outputs[i]
            best_patch = patches_l[i]
            best_coords = coords_l[i]

    return best_patch, best_coords

# Load the pre-trained ResNet50 model
model_path = 'C://Users//Ronve//PycharmProjects//THE_Project//PyTorch_train//resnet50_custom_dataset.pth'
model = load_model(model_path, 4)

# Define parameters
base_folder = 'C://Users//Ronve//PycharmProjects//THE_Project//samplingFillMean//sampling_50_percent'
num_patches = 4
overlap = 0.5
output_folder = 'C://Users//Ronve//PycharmProjects//THE_Project//best_patches//Test'
os.makedirs(output_folder, exist_ok=True)

best_patches_info = []

# Process each image in the base folder
for class_folder in os.listdir(base_folder):
    class_folder_path = os.path.join(base_folder, class_folder)
    if os.path.isdir(class_folder_path):
        for img_name in os.listdir(class_folder_path):
            img_path = os.path.join(class_folder_path, img_name)
            image = Image.open(img_path).convert('RGB')

            # Get patches
            patches = get_patches(image, num_patches, overlap)

            # Classify patches and find the best one
            best_patch, best_coords = classify_patches(model, patches)

            # Save the best patch
            output_path = os.path.join(output_folder, class_folder)
            os.makedirs(output_path, exist_ok=True)
            torchvision.utils.save_image(best_patch, os.path.join(output_path, img_name))

            # Store the best patch info
            best_patches_info.append({'image_path': img_path, 'coords': best_coords})

# Convert the list of dictionaries to a DataFrame and save to an Excel file
df = pd.DataFrame(best_patches_info)
df[['x', 'y']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
df.drop(columns=['coords'], inplace=True)
df.to_excel('best_patches_info.xlsx', index=False)

print("Processing completed!")
