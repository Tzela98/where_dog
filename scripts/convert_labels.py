import os
from PIL import Image

# Paths to the dataset
images_dir = 'dataset/images/train'
labels_dir = 'dataset/labels/train'
labels_corrected_dir = 'dataset/labels_corrected/train'

# Create the corrected labels directory if it doesn't exist
os.makedirs(labels_corrected_dir, exist_ok=True)

# Function to get image dimensions
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

# Iterate over all annotation files
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        # Get the corresponding image path
        image_name = filename.replace('.txt', '.jpg')  # Assuming images are .jpg
        image_path = os.path.join(images_dir, image_name)
        
        # Get image dimensions
        if os.path.exists(image_path):
            image_width, image_height = get_image_dimensions(image_path)
        else:
            print(f"Image not found: {image_path}")
            continue
        
        # Read the annotation file
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Fix the annotations
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] == 'dog':
                # Convert class name to class ID
                class_id = 0
                
                # Convert absolute coordinates to normalized coordinates
                x_min = float(parts[1])
                y_min = float(parts[2])
                x_max = float(parts[3])
                y_max = float(parts[4])
                
                x_center = (x_min + x_max) / (2 * image_width)
                y_center = (y_min + y_max) / (2 * image_height)
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                # Write the corrected annotation
                new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
            else:
                new_lines.append(line)
        
        # Write the corrected annotations to the new directory
        corrected_filepath = os.path.join(labels_corrected_dir, filename)
        with open(corrected_filepath, 'w') as f:
            f.writelines(new_lines)

print("Annotations corrected and saved in the new directory.")