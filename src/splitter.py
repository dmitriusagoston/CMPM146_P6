import random
import os
import shutil
from config import train_directory, test_directory, train_size, categories


def split_dataset(source_directory, train_target_directory, test_target_directory, categories=None, train_ratio=0.8):
    if categories is None:
        categories = os.listdir(source_directory)
    
    for category in categories:
        source_category_dir = os.path.join(source_directory, category)
        train_category_dir = os.path.join(train_target_directory, category)
        test_category_dir = os.path.join(test_target_directory, category)
        
        if not os.path.exists(train_category_dir):
            os.makedirs(train_category_dir)
        if not os.path.exists(test_category_dir):
            os.makedirs(test_category_dir)
        
        image_files = os.listdir(source_category_dir)
        random.shuffle(image_files)  # Shuffle the images
        
        train_count = int(len(image_files) * train_ratio)  # Number of images in the training set
        train_files = image_files[:train_count]
        test_files = image_files[train_count:]
        
        # Copy the images to the train directory
        for image_file in train_files:
            source_image_path = os.path.join(source_category_dir, image_file)
            target_image_path = os.path.join(train_category_dir, image_file)
            shutil.copy(source_image_path, target_image_path)
        
        # Copy the images to the test directory
        for image_file in test_files:
            source_image_path = os.path.join(source_category_dir, image_file)
            target_image_path = os.path.join(test_category_dir, image_file)
            shutil.copy(source_image_path, target_image_path)

if __name__ == "__main__":
    source_directory = 'kaggle/doom_or_ac'
    train_target_directory = 'kaggle/train_d_or_ac'
    test_target_directory = 'kaggle/test_d_or_ac'

    split_dataset(source_directory, train_target_directory, test_target_directory, categories)
    print('Train and test datasets have been extracted.')