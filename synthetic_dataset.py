import os
import numpy as np
import cv2
import random

# Create dataset directories
dataset_path = "synthetic_dataset"
image_path = os.path.join(dataset_path, "images", "val")
label_path = os.path.join(dataset_path, "labels", "val")

os.makedirs(image_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)

# Dataset parameters
num_images = 10
image_size = (512, 512)
class_id = 0

def create_dataset():
    # Create YAML config file
    with open(os.path.join(dataset_path, "dataset.yaml"), "w") as f:
        f.write(f"""path: {dataset_path}
train: images/train
train_labels: labels/train
val: images/train  # Using same data for validation
val_labels: labels/train

names:
  0: rectangle
""")
    
    for i in range(num_images):
        # Create black background image (16-bit)
        img = np.zeros((image_size[0], image_size[1]), dtype=np.uint16)
        
        # Determine number of rectangles for this image (1 to 5)
        num_rectangles = random.randint(1, 5)
        
        # Keep track of existing rectangles to avoid overlap
        existing_rects = []
        
        # Create annotation file
        annotation_file = os.path.join(label_path, f"image_{i:04d}.txt")
        with open(annotation_file, "w") as f:
            
            # Add rectangles
            for _ in range(num_rectangles):
                # Try to place a non-overlapping rectangle (max 20 attempts)
                for attempt in range(20):
                    # Random rectangle size (5% to 20% of image size)
                    w = random.uniform(0.05, 0.2)
                    h = random.uniform(0.05, 0.2)
                    
                    # Random center position
                    cx = random.uniform(w/2, 1-w/2)
                    cy = random.uniform(h/2, 1-h/2)
                    
                    # Calculate pixel coordinates
                    x1 = int((cx - w/2) * image_size[1])
                    y1 = int((cy - h/2) * image_size[0])
                    x2 = int((cx + w/2) * image_size[1])
                    y2 = int((cy + h/2) * image_size[0])
                    
                    # Check for overlap with existing rectangles
                    overlap = False
                    for rect in existing_rects:
                        r_x1, r_y1, r_x2, r_y2 = rect
                        if (x1 < r_x2 and x2 > r_x1 and 
                            y1 < r_y2 and y2 > r_y1):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Add to existing rectangles
                        existing_rects.append((x1, y1, x2, y2))
                        
                        # Draw white rectangle on image (16-bit white = 65535)
                        img[y1:y2, x1:x2] = 65535
                        
                        # Write YOLO format annotation: class_id cx cy w h
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        break
            
        # Save image as 16-bit PNG
        cv2.imwrite(os.path.join(image_path, f"image_{i:04d}.png"), img, 
                   [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        print(f"Created image {i+1}/{num_images}")

if __name__ == "__main__":
    create_dataset()