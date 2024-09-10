"""
This file contains functions for plotting data and model results.
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from torch import nn


def identify_imgs(model: nn.Module,
                  base_folder: str,
                  class_names: list,
                  transforms: torchvision.transforms.Compose,
                  device: torch.device = 'cpu'):
    """
    Takes a trained model, a base folder path, a list of class names, transforms, and a device.
    Selects one random image from each subfolder, processes it, and creates a plot of the original
    and processed images with predictions.

    Args:
    model (nn.Module): The trained model
    base_folder (str): Path to the base folder (e.g., 'train' or 'test') containing class subfolders
    class_names (list): List of class names
    transforms (torchvision.transforms.Compose): Image transforms to apply
    device (torch.device): Device to run the model on (default: 'cpu')

    Example usage:
    identify_imgs(model=model_0,
                  base_folder='./data/test',
                  class_names=class_names,
                  transforms=data_transform,
                  device=device)
    """
    # Get all subfolders (classes)
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    
    # Set up the plot
    fig, axes = plt.subplots(len(subfolders), 2, figsize=(12, 7*len(subfolders)))
    fig.suptitle("Sample Images and Predictions from Each Class", fontsize=16)
    
    # Ensure axes is always 2D
    if len(subfolders) == 1:
        axes = axes.reshape(1, -1)
    
    plt.subplots_adjust(hspace=0.5)

    model.eval()
    
    for idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(base_folder, subfolder)
        
        # Get all image files in the subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # Select a random image
            random_image = random.choice(image_files)
            img_path = os.path.join(subfolder_path, random_image)
            
            # Read and process the image
            raw_image = torchvision.io.read_image(img_path).float() / 255.0  # Normalize to [0, 1]
            img = transforms(raw_image)
            
            # Make prediction
            with torch.inference_mode():
                probabilities = model(img.unsqueeze(0).to(device)).softmax(dim=1)
                class_index = probabilities.argmax(dim=1)
                label = class_names[class_index]
                percentage = probabilities[0][class_index].item() * 100
            
            # Display original image
            axes[idx, 0].imshow(raw_image.permute(1, 2, 0))
            axes[idx, 0].set_title('Original')
            axes[idx, 0].axis('off')
            
            # Display processed image
            axes[idx, 1].imshow(img.permute(1, 2, 0).clamp(0, 1))  # Clamp values to [0, 1]
            axes[idx, 1].set_title('Transformed')
            axes[idx, 1].axis('off')
            
            # Add text annotation above the subplot
            axes[idx, 0].text(0.5, 1.1, f'True: {subfolder}', ha='center', va='bottom', transform=axes[idx, 0].transAxes)
            axes[idx, 1].text(0.5, 1.1, f'Pred: {label} (Certainty: {percentage:.2f}%)', ha='center', va='bottom', transform=axes[idx, 1].transAxes)
        else:
            axes[idx, 0].text(0.5, 0.5, "No images found", ha='center', va='center')
            axes[idx, 0].axis('off')
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_acc_and_loss(results_dict, model: nn.Module):
    """
    Takes a dictionary of results and plots the accuracy and loss curves.

    Args:
    results_dict: A dictionary of losses and accuracies collected 
                  during model training/testing.
    model: A target PyTorch model 

    Example usage:
    plot_acc_and_loss(results_dict=results, model=model_0)
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    num_epochs = len(results_dict.get('train_acc'))

    fig.suptitle(f'Model metrics: {model.__class__.__name__}')
    
    ax1.plot(range(1, num_epochs+1), results_dict.get('train_acc'), label='train')
    ax1.plot(range(1, num_epochs+1), results_dict.get('test_acc'), label='test')
    ax1.set_title('Accuracy Train/Test')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(range(1, num_epochs+1), results_dict.get('train_loss'), label='train')
    ax2.plot(range(1, num_epochs+1), results_dict.get('test_loss'), label='test')
    ax2.set_title('Loss Train/Test')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    
    plt.tight_layout()
    plt.show()

def show_sample_images(folder_path, num_cols=3):
    """
    Display one random image from each subfolder in the given folder path.
    
    Args:
    folder_path (str): Path to the main folder (train or test) containing class subfolders
    num_cols (int): Number of columns in the plot grid (default: 3)
    
    Returns:
    None

    Example:
    show_sample_images('path/to/your/train/folder')
    or
    show_sample_images('path/to/your/test/folder')
    """
    # Get all subfolders (classes)
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Calculate the number of rows needed
    num_rows = (len(subfolders) + num_cols - 1) // num_cols
    
    # Set up the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    fig.suptitle("Sample Images from Each Class", fontsize=16)
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten() if num_rows > 1 else [axes]
    
    for idx, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # Get all image files in the subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # Select a random image
            random_image = random.choice(image_files)
            img_path = os.path.join(subfolder_path, random_image)
            
            # Open and display the image
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(subfolder)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, "No images found", ha='center', va='center')
            axes[idx].axis('off')
    
    # Remove any unused subplots
    for idx in range(len(subfolders), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

