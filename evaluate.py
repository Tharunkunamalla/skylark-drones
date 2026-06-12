import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import get_train_val_split, GCPDataset, CLASS_MAPPING, REV_CLASS_MAPPING
from transforms import get_val_transforms
from model import GCPMultitaskModel

def run_evaluation(base_dir, json_path, model_weights_path, img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get validation keys
    _, val_keys, json_data = get_train_val_split(json_path, base_dir, val_split=0.2)
    
    # Load transforms and dataset
    val_tf = get_val_transforms(img_size=img_size)
    dataset = GCPDataset(base_dir, val_keys, json_data, transform=val_tf)
    
    # Initialize model
    print("Loading model weights...")
    model = GCPMultitaskModel(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Track metrics
    pixel_errors = []
    all_class_targets = []
    all_class_preds = []

    print("\nRunning evaluation on validation set...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            key = val_keys[idx]
            val = json_data[key]
            
            # Original image shape
            img_path = os.path.join(base_dir, key)
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
                
            # Ground truth in original resolution space
            orig_x_gt = val["mark"]["x"]
            orig_y_gt = val["mark"]["y"]
            shape_gt = CLASS_MAPPING[val["verified_shape"]]

            # Model input
            sample = dataset[idx]
            image_tensor = sample["image"].unsqueeze(0).to(device) # Shape: [1, 3, H, W]

            # Forward pass
            coord_pred, class_logits = model(image_tensor)
            
            # Extract outputs
            pred_x_norm = coord_pred[0, 0].item()
            pred_y_norm = coord_pred[0, 1].item()
            
            # Convert predicted normalized coords back to original resolution space
            pred_x_orig = pred_x_norm * orig_w
            pred_y_orig = pred_y_norm * orig_h
            
            # Compute Euclidean distance (pixel error) in original resolution space
            dist = np.sqrt((pred_x_orig - orig_x_gt)**2 + (pred_y_orig - orig_y_gt)**2)
            pixel_errors.append(dist)
            
            # Classification pred
            class_pred = torch.argmax(class_logits, dim=1).item()
            all_class_preds.append(class_pred)
            all_class_targets.append(shape_gt)

    pixel_errors = np.array(pixel_errors)
    
    # Calculate metrics
    mean_pixel_error = np.mean(pixel_errors)
    median_pixel_error = np.median(pixel_errors)
    
    pck_10 = np.mean(pixel_errors <= 10.0) * 100.0
    pck_25 = np.mean(pixel_errors <= 25.0) * 100.0
    pck_50 = np.mean(pixel_errors <= 50.0) * 100.0
    
    accuracy = accuracy_score(all_class_targets, all_class_preds)
    macro_f1 = f1_score(all_class_targets, all_class_preds, average="macro")
    
    # Generate report
    report_text = []
    report_text.append("=" * 60)
    report_text.append("        AERIAL GCP MULTITASK BASELINE EVALUATION REPORT")
    report_text.append("=" * 60)
    report_text.append(f"Model Checkpoint: {model_weights_path}")
    report_text.append(f"Validation Samples: {len(val_keys)}")
    report_text.append(f"Resized Input Size: {img_size}x{img_size}")
    report_text.append("-" * 60)
    report_text.append("KEYPOINT LOCALIZATION METRICS (in original pixel resolution):")
    report_text.append(f"  - Mean Pixel Error:   {mean_pixel_error:.2f} px")
    report_text.append(f"  - Median Pixel Error: {median_pixel_error:.2f} px")
    report_text.append(f"  - PCK@10 Accuracy:     {pck_10:.2f} %")
    report_text.append(f"  - PCK@25 Accuracy:     {pck_25:.2f} %")
    report_text.append(f"  - PCK@50 Accuracy:     {pck_50:.2f} %")
    report_text.append("-" * 60)
    report_text.append("SHAPE CLASSIFICATION METRICS:")
    report_text.append(f"  - Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    report_text.append(f"  - Macro F1-Score:     {macro_f1:.4f}")
    report_text.append("\nDetailed Classification Report:")
    
    target_names = [REV_CLASS_MAPPING[i] for i in range(3)]
    detailed_report = classification_report(all_class_targets, all_class_preds, target_names=target_names, zero_division=0)
    report_text.append(detailed_report)
    report_text.append("=" * 60)
    
    report_string = "\n".join(report_text)
    
    print("\n" + report_string)
    
    # Save to file
    report_save_path = "evaluation_report.txt"
    with open(report_save_path, "w") as f:
        f.write(report_string)
    print(f"Report saved to {report_save_path}")

if __name__ == "__main__":
    JSON_PATH = os.path.join("train_dataset", "train_dataset", "gcp_marks.json")
    BASE_DIR = os.path.join("train_dataset", "train_dataset")
    WEIGHTS_PATH = "best_model.pth"
    
    run_evaluation(BASE_DIR, JSON_PATH, WEIGHTS_PATH)
