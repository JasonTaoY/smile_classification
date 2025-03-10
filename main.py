from model.load import initialize_model
from data.load import FaceDataset, get_deepface_pred, get_data
from model.trainer import train
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

model,loss,opt = initialize_model(device="cuda")
labelled_image, non_labelled_image, label, test_folders = get_data()
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

labeled_dataset = FaceDataset(labelled_image, labels=label, transform=train_transform)
unlabeled_dataset = FaceDataset(non_labelled_image, labels=None, transform=train_transform)
labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)

non_labelled_image_deepface = get_deepface_pred(non_labelled_image)

total_train_loss, total_val_loss = train(model, loss, opt, labeled_loader, unlabeled_loader, non_labelled_image, label, labelled_image,
      None, train_transform, val_transform, "cuda")


train_loss_list = total_train_loss

valid_loss_list = total_val_loss

plt.figure(figsize=(10, 6))

epoch_counter = 0
epoch_ticks = []
epoch_labels = []

for i, (train_losses, valid_losses) in enumerate(zip(train_loss_list, valid_loss_list)):
    epochs = len(train_losses)
    epoch_range = np.arange(1, epochs + 1)

    plt.plot(epoch_range, train_losses, label=f'Train Loss - Round {i+1}')

    if valid_losses:
        plt.plot(epoch_range, valid_losses, linestyle='--', label=f'Valid Loss - Round {i+1}')

    epoch_ticks.extend(epoch_range)
    epoch_labels.extend([f'E{j+1}' for j in range(epochs)])

plt.xticks(epoch_ticks, epoch_labels, rotation=45)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Per Round')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

