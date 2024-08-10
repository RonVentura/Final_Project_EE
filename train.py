import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

def train_resnet50(train_data_dir, val_data_dir):
    # Define transformation for the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets with PyTorch ImageFolder and set class labels
    image_datasets = {x: datasets.ImageFolder(os.path.join(train_data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    # Get the number of classes and class labels
    num_classes = len(image_datasets['train'].classes)
    class_labels = image_datasets['train'].classes

    # Load dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    # Load the pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully-connected layer with a new one
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    def train_model(model, criterion, optimizer, num_epochs=25):
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += labels.size(0)

                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # Record loss and accuracy for plotting
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        # Plot training and validation accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

        return model

    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trained_model = train_model(model, criterion, optimizer, num_epochs=10)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'pythorch_model_50_per.pth')

if __name__ == '__main__':
    freeze_support()
    train_resnet50('C://Users//Ronve//PycharmProjects//THE_Pr oject//TrainOnSampled//sampling_50', 'C://Users//Ronve//PycharmProjects//THE_Project//TrainOnSampled//sampling_50')
