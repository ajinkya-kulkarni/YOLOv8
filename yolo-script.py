#!/usr/bin/env python3

import os
import random
import glob
import yaml
import shutil
import argparse
from collections import defaultdict
import re
from typing import List, Dict, Tuple
from tqdm import tqdm

# ============================================================================
# Argument Parsing
# ============================================================================

def parse_arguments():
    """
    Parse command line arguments for dataset processing configuration.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Process cell training data and convert to YOLO format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help='Directory where processed dataset will be saved'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--marker-mode',
        type=str,
        choices=['include', 'exclude'],
        default='include',
        help='Whether to include or exclude specified markers'
    )
    
    parser.add_argument(
        '--markers',
        type=str,
        nargs='+',  # Accept one or more markers
        default=['CD3'],
        help='List of markers to include/exclude (based on marker-mode)'
    )
    
    parser.add_argument(
        '--min-boxes',
        type=int,
        default=1,
        help='Minimum number of bounding boxes required for an image to be included'
    )
    
    args = parser.parse_args()
    return args

# ============================================================================
# Configuration Constants
# ============================================================================

# Directory Configuration
IMDIR = "/home/ajinkya.kulkarni/Ultivue_Data/cell_training_data/CellDataTrain/train/imdir"
ANNDIR = "/home/ajinkya.kulkarni/Ultivue_Data/cell_training_data/CellDataTrain/train/anndir"

# Dataset Parameters
TRAIN_SPLIT = 0.8
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
RANDOM_SEED = 123

# ============================================================================
# Utility Functions
# ============================================================================

def get_known_markers(anndir: str) -> List[str]:
    """
    Extract marker names from annotation filenames.
    Example: annotations_CD4.txt -> returns "CD4"
    """
    pattern = os.path.join(anndir, "annotations_*.txt")
    markers = {os.path.basename(f)[len("annotations_"):-4] for f in glob.glob(pattern)}
    return sorted(markers)

def filter_images_by_markers(all_images: List[str], markers: List[str], include: bool) -> List[str]:
    """
    Filter images based on marker patterns.
    Args:
        all_images: List of all image filenames
        markers: List of marker patterns to search for
        include: If True, keep files containing markers. If False, exclude them
    Returns:
        List of filtered image filenames
    """
    if not markers:
        return all_images
    
    print(f"Filtering images - Mode: {'Include' if include else 'Exclude'}")
    print(f"Markers: {', '.join(markers)}")
    
    pattern = re.compile('|'.join(markers))
    filtered_images = []
    
    for img in all_images:
        has_marker = bool(pattern.search(img))
        if (include and has_marker) or (not include and not has_marker):
            filtered_images.append(img)
    
    print(f"Kept {len(filtered_images)} images after filtering\n")
    return filtered_images

def find_marker_in_filename(filename: str, markers: List[str]) -> str:
    """
    Check if filename contains any marker pattern.
    Returns the marker name if found, None otherwise.
    """
    for marker in markers:
        if f"_{marker}_" in filename:
            return marker
    return None

# ============================================================================
# Data Processing Functions
# ============================================================================

def parse_annotation_files(anndir: str, valid_images: List[str]) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """
    Read all annotation files and collect bounding boxes.
    Returns a dictionary mapping image names to lists of bounding boxes (x, y, width, height).
    """
    image_to_bboxes = defaultdict(list)
    valid_set = set(valid_images)
    
    for annot_path in glob.glob(os.path.join(anndir, "annotations_*.txt")):
        with open(annot_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                parts = line.strip().split(":")
                if len(parts) != 2:
                    continue

                img_name, bbox_str = parts
                img_name = img_name.strip()
                
                # Only process annotations for images in our filtered set
                if img_name not in valid_set:
                    continue

                # Parse each bounding box token
                for token in bbox_str.strip().split():
                    try:
                        x1, y1, x2, y2 = map(int, token.split(","))
                        w = x2 - x1
                        h = y2 - y1
                        if w > 0 and h > 0:  # Ensure valid box dimensions
                            image_to_bboxes[img_name].append((x1, y1, w, h))
                    except ValueError:
                        continue
                        
    return image_to_bboxes

def filter_images_by_box_count(images: List[str], bboxes: Dict[str, List[Tuple]], min_boxes: int) -> List[str]:
    """
    Filter images to only include those with at least min_boxes valid bounding boxes.
    """
    filtered = []
    for img in images:
        if len(bboxes.get(img, [])) >= min_boxes:
            filtered.append(img)
            
    removed = len(images) - len(filtered)
    if removed > 0:
        print(f"Removed {removed} images with fewer than {min_boxes} bounding boxes")
        
    return filtered

def create_yolo_labels(images_list: List[str], bboxes_dict: Dict[str, List[Tuple]], output_dir: str, subset: str) -> None:
    """
    Create YOLO format annotation files.
    
    Args:
        images_list: List of image filenames
        bboxes_dict: Dictionary mapping image names to bounding boxes (x, y, w, h)
        output_dir: Base output directory
        subset: 'train' or 'val'
    """
    # Create labels directory using the structure from your working dataset
    labels_dir = os.path.join(output_dir, "labels", subset)
    os.makedirs(labels_dir, exist_ok=True)
    
    for img_name in tqdm(images_list, desc=f"Creating {subset} labels"):
        # Get bounding boxes for this image
        boxes = bboxes_dict.get(img_name, [])
        
        # Skip empty boxes - shouldn't happen now with filtering
        if not boxes:
            continue
        
        # Create annotation file (use same name but .txt extension)
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        with open(txt_path, "w") as f:
            for x, y, w, h in boxes:
                # Convert to YOLO format:
                # - Class ID is always 0 (single class)
                # - Convert to center coordinates
                # - Normalize by image dimensions
                x_center = (x + w/2) / IMAGE_WIDTH
                y_center = (y + h/2) / IMAGE_HEIGHT
                w_norm = w / IMAGE_WIDTH
                h_norm = h / IMAGE_HEIGHT
                
                # Ensure values are within bounds [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))
                
                # Write YOLO format line: class_id x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def create_data_yaml(output_dir: str) -> None:
    """
    Create a YOLO data.yaml configuration file.
    
    Args:
        output_dir: Base output directory
    """
    # Use relative paths like in your working example
    data_yaml = {
        "path": ".",  # Use relative path
        "train": "images/train",
        "train_labels": "labels/train",
        "val": "images/val",
        "val_labels": "labels/val",
        "names": {
            0: "positive"  # Single class dataset
        }
    }
    
    # Write to file
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created dataset configuration at: {yaml_path}")

def copy_image_files(images: List[str], src_dir: str, dst_dir: str) -> None:
    """
    Copy images from source to destination directory.
    Creates the destination directory if it doesn't exist.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for img in tqdm(images, desc=f"Copying images to {os.path.basename(dst_dir)}"):
        src = os.path.join(src_dir, img)
        dst = os.path.join(dst_dir, img)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: Source file not found: {src}")

# ============================================================================
# Main Processing
# ============================================================================

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducible splits
    random.seed(RANDOM_SEED)
    
    # Step 1: Get all available markers and images
    known_markers = get_known_markers(ANNDIR)
    print("\nFound markers:", known_markers)
    print()
    
    all_images = os.listdir(IMDIR)
    print(f"Total images found: {len(all_images)}\n")
    
    # Step 2: Filter images based on command-line arguments
    include_mode = args.marker_mode.lower() == "include"
    filtered_images = filter_images_by_markers(all_images, args.markers, include_mode)
    if not filtered_images:
        print("Error: No images to process after filtering. Exiting.")
        return
        
    # Step 3: Parse annotations and filter images that don't have enough boxes
    bboxes = parse_annotation_files(ANNDIR, filtered_images)
    
    if args.min_boxes > 0:
        filtered_images = filter_images_by_box_count(filtered_images, bboxes, args.min_boxes)
        if not filtered_images:
            print("Error: No images with sufficient annotations. Try lowering --min-boxes value.")
            return
    
    # Step 4: Group images by marker for balanced splitting
    marker_images = defaultdict(list)
    for img in filtered_images:
        marker = find_marker_in_filename(img, known_markers)
        if marker:
            marker_images[marker].append(img)
    
    # Step 5: Split into train/val while preserving marker distribution
    train_images = []
    val_images = []
    
    for marker, images in marker_images.items():
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)
        train_images.extend(images[:split_idx])
        val_images.extend(images[split_idx:])
        print(f"Marker '{marker}': {len(images)} total -> {split_idx} train, {len(images)-split_idx} val")
    
    print(f"\nTotal: {len(train_images)} train, {len(val_images)} val images")
    print()
    
    # Step 6: Create output directory structure using desired format
    base_dir = args.outdir
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory structure to match your working example
    train_images_dir = os.path.join(base_dir, "images", "train")
    val_images_dir = os.path.join(base_dir, "images", "val")
    
    # Step 7: Copy images to their directories
    copy_image_files(train_images, IMDIR, train_images_dir)
    copy_image_files(val_images, IMDIR, val_images_dir)
    
    # Step 8: Create YOLO annotation files
    print("Creating YOLO annotations...")
    create_yolo_labels(train_images, bboxes, base_dir, "train")
    create_yolo_labels(val_images, bboxes, base_dir, "val")
    
    # Step 9: Create data.yaml configuration file
    create_data_yaml(base_dir)
    
    print("\nDone! YOLO dataset created successfully!")
    print(f"Dataset location: {os.path.abspath(base_dir)}\n")

if __name__ == "__main__":
    main()