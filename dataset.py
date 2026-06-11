import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

CLASS_MAPPING = {
    "Cross": 0,
    "Square": 1,
    "L-Shape": 2
}
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

class GCPDataset(Dataset):
    """
    PyTorch Dataset for Aerial GCP Pose Estimation and Shape Classification.
    """
    def __init__(self, base_dir, keys, json_data, transform=None):
        self.base_dir = base_dir
        self.keys = keys
        self.json_data = json_data
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        val = self.json_data[key]
        
        # Load image
        img_path = os.path.join(self.base_dir, key)
        # Using PIL to open and converting to numpy array
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Keypoints and shape class
        x = val["mark"]["x"]
        y = val["mark"]["y"]
        shape_name = val["verified_shape"]
        shape_class = CLASS_MAPPING[shape_name]

        # Apply transforms if provided
        if self.transform:
            # Check if using Albumentations
            if hasattr(self.transform, '__module__') and 'albumentations' in self.transform.__module__:
                transformed = self.transform(image=image, keypoints=[(x, y)])
                image = transformed["image"]
                if len(transformed["keypoints"]) > 0:
                    x, y = transformed["keypoints"][0]
            else:
                # Fallback to basic torchvision-like transform (or custom callable)
                image = self.transform(image)
        else:
            # Default to converting image to float tensor normalized to [0, 1]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return {
            "image": image,
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "shape_class": torch.tensor(shape_class, dtype=torch.long)
        }

def get_train_val_split(json_path, base_dir, val_split=0.2, seed=42):
    """
    Reads the JSON, filters out missing files and missing class labels,
    and returns stratified train and validation lists of keys.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter keys based on file existence and valid shape annotation
    valid_keys = []
    labels = []
    
    for key, val in data.items():
        full_path = os.path.join(base_dir, key)
        if os.path.exists(full_path) and "verified_shape" in val and val["verified_shape"] in CLASS_MAPPING:
            valid_keys.append(key)
            labels.append(CLASS_MAPPING[val["verified_shape"]])
            
    print(f"Dataset Split Utility:")
    print(f"  - Labeled images in JSON: {len(data)}")
    print(f"  - Valid and physically existing images: {len(valid_keys)}")

    # Perform stratified train/validation split
    train_keys, val_keys = train_test_split(
        valid_keys,
        test_size=val_split,
        random_state=seed,
        stratify=labels
    )
    
    print(f"  - Training samples: {len(train_keys)}")
    print(f"  - Validation samples: {len(val_keys)}")
    
    return train_keys, val_keys, data

def get_dataloaders(base_dir, json_path, batch_size=8, val_split=0.2, train_transform=None, val_transform=None, seed=42):
    """
    Helper function to create Train and Validation DataLoaders.
    """
    train_keys, val_keys, json_data = get_train_val_split(json_path, base_dir, val_split, seed)

    train_dataset = GCPDataset(base_dir, train_keys, json_data, transform=train_transform)
    val_dataset = GCPDataset(base_dir, val_keys, json_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

from transforms import get_train_transforms, get_val_transforms

if __name__ == "__main__":
    # Test execution
    JSON_PATH = os.path.join("train_dataset", "train_dataset", "gcp_marks.json")
    BASE_DIR = os.path.join("train_dataset", "train_dataset")

    # Define transforms
    img_size = 512
    train_tf = get_train_transforms(img_size=img_size)
    val_tf = get_val_transforms(img_size=img_size)

    print("Creating DataLoaders with Albumentations transforms...")
    train_loader, val_loader = get_dataloaders(
        BASE_DIR, 
        JSON_PATH, 
        batch_size=4, 
        train_transform=train_tf, 
        val_transform=val_tf
    )
    
    # Load one batch
    batch = next(iter(train_loader))
    print("\nVerifying DataLoader sample batch:")
    print(f"  - Image batch shape: {batch['image'].shape} (type: {batch['image'].dtype})")
    print(f"  - Keypoint x shape: {batch['x'].shape} (type: {batch['x'].dtype})")
    print(f"  - Keypoint y shape: {batch['y'].shape} (type: {batch['y'].dtype})")
    print(f"  - Shape class shape: {batch['shape_class'].shape} (type: {batch['shape_class'].dtype})")
    print(f"  - Classes in batch: {batch['shape_class'].tolist()} -> {[REV_CLASS_MAPPING[cid] for cid in batch['shape_class'].tolist()]}")

    # Verify keypoint scaling on a single sample
    train_keys, _, json_data = get_train_val_split(JSON_PATH, BASE_DIR, val_split=0.2)
    sample_key = train_keys[0]
    sample_val = json_data[sample_key]
    
    # Get original image dims
    img_path = os.path.join(BASE_DIR, sample_key)
    with Image.open(img_path) as img:
        orig_w, orig_h = img.size
        
    orig_x = sample_val["mark"]["x"]
    orig_y = sample_val["mark"]["y"]
    
    # Load sample via Dataset
    dataset_sample = GCPDataset(BASE_DIR, [sample_key], json_data, transform=train_tf)[0]
    scaled_x = dataset_sample["x"].item()
    scaled_y = dataset_sample["y"].item()
    
    expected_x = orig_x * (img_size / orig_w)
    expected_y = orig_y * (img_size / orig_h)
    
    print("\nVerifying GCP Keypoint Scaling Correctness:")
    print(f"  - Original image shape: {orig_w}x{orig_h}")
    print(f"  - Target image shape: {img_size}x{img_size}")
    print(f"  - Original keypoint coords: ({orig_x:.3f}, {orig_y:.3f})")
    print(f"  - Scaled keypoint coords:   ({scaled_x:.3f}, {scaled_y:.3f})")
    print(f"  - Expected scaled coords:   ({expected_x:.3f}, {expected_y:.3f})")
    
    # Assert they are close (allowing floating point delta)
    assert abs(scaled_x - expected_x) < 1e-2, f"X scaling error: got {scaled_x}, expected {expected_x}"
    assert abs(scaled_y - expected_y) < 1e-2, f"Y scaling error: got {scaled_y}, expected {expected_y}"
    print("  => GCP keypoint scaling is mathematically verified and correct!")
