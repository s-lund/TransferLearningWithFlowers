"""
Some functions to preprocess the data to make the notebooks cleaner.
"""


import os
import random
import shutil


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#NUM_WORKERS = int(os.cpu_count() / 2)
NUM_WORKERS = 0

def create_dataloader(
    dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    shuffle: bool = True
):
    """
    Creates DataLoaders from PyTorch datasets.
    Because of the need for different transforms for training and
    testing this needs to be called twice.

    Args:
        dir: Directory path to the data.
        transform: torchvision transforms to be applied to the data.
        batch_size: Number of samples per batch.
        num_workers: Number of workers for the DataLoader.
        shuffle: Whether to shuffle the data. 
            (Should be True for training and False for testing)

    Returns:
        A tuple of one DataLoader and class_names.
        Classs_names is a list of the target classes.

    Example:
        train_dataloader, class_names = create_dataloaders(
            dir="pass/to/train_dir",
            transform=data_transforms,
            batch_size=32,
            num_workers=4,
        )

    """
    data = datasets.ImageFolder(root=dir,
                                transform=transform, # transform for the data
                                target_transform=None)    # transform for the label/target

    dataloader = DataLoader(dataset=data, 
                               batch_size=batch_size, 
                               shuffle=True, 
                               num_workers=num_workers,
                               pin_memory=True)

    return dataloader, data.classes


def split_train_test(source_dir, test_ratio=0.2):
    """
    Split the data in the source directory into train and test sets.
    The test set will be moved to a new directory called 'test' in the parent directory of the source directory.
    If the test directory already exists, it will be skipped.
    
    Args:
    source_dir (str): Path to the source directory containing class subdirectories
    test_ratio (float): Ratio of images to move to the test set (default: 0.2)
    
    Returns:
    None

    Usage:
    source_directory = 'data/flowers/train'
    split_train_test(source_directory)
    """
    # Get the parent directory of the source directory
    parent_dir = os.path.dirname(source_dir)
    
    # Create the test directory
    test_dir = os.path.join(parent_dir, 'test')

    # Check if the test directory already exists
    if os.path.exists(test_dir):
        print(f"Test directory already exists. Skipping split operation.")
        return

    os.makedirs(test_dir, exist_ok=True)
    
    # Iterate through each class subdirectory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        print(f'Moving {class_name} from {class_dir} to {test_dir}')
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Create corresponding test subdirectory
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all image files
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Calculate number of images to move
        num_test = int(len(images) * test_ratio)
        
        # Randomly select images to move
        test_images = random.sample(images, num_test)
        
        # Move selected images to test directory
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_class_dir, img)
            shutil.move(src, dst)
        
        print(f"Moved {num_test} images from {class_name} to test set")

