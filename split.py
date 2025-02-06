import os
import shutil
import random

# Define directories
base_dir = "Images/"
train_dir = "images_split/train"
val_dir = "images_split/validation"
test_dir = "images_split/test"

# Create destination folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(base_dir):
    class_folder = os.path.join(base_dir, class_name)
    
    if os.path.isdir(class_folder):
        # Create class subdirectories in train, validation, and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Filter only image files
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"⚠️ Warning: No images found in {class_folder}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Split into train (80%), val (10%), test (10%)
        total_images = len(images)
        train_split = int(0.8 * total_images)
        val_split = int(0.9 * total_images)

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        # Copy files to new directories
        for image in train_images:
            src = os.path.join(class_folder, image)
            dst = os.path.join(train_dir, class_name, image)
            shutil.copy(src, dst)

        for image in val_images:
            src = os.path.join(class_folder, image)
            dst = os.path.join(val_dir, class_name, image)
            shutil.copy(src, dst)

        for image in test_images:
            src = os.path.join(class_folder, image)
            dst = os.path.join(test_dir, class_name, image)
            shutil.copy(src, dst)

        print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("🎉 Data split complete!")
