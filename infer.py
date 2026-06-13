import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transforms import get_val_transforms
from model import GCPMultitaskModel

SHAPE_CLASSES = {
    0: "Cross",
    1: "Square",
    2: "L-Shaped"
}

def find_test_images(test_dir):
    """
    Finds all image files (.jpg, .jpeg, .png) recursively under the test directory.
    Returns lists of absolute paths and relative paths.
    """
    abs_paths = []
    rel_paths = []
    
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, test_dir)
                # Ensure path separator is forward slash for consistency in JSON
                rel_path = rel_path.replace("\\", "/")
                abs_paths.append(abs_path)
                rel_paths.append(rel_path)
                
    return abs_paths, rel_paths

def main():
    TEST_DIR = os.path.join("test_dataset", "test_dataset")
    WEIGHTS_PATH = "best_model.pth"
    OUTPUT_JSON = "predictions.json"
    IMG_SIZE = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Scanning test dataset directory: {TEST_DIR}...")
    abs_paths, rel_paths = find_test_images(TEST_DIR)
    print(f"Found {len(abs_paths)} test images.")

    if len(abs_paths) == 0:
        print("Error: No test images found.")
        return

    val_tf = get_val_transforms(img_size=IMG_SIZE)

    print("Loading model weights...")
    model = GCPMultitaskModel(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = {}

    print("Running inference on test dataset...")
    with torch.no_grad():
        for abs_path, rel_path in tqdm(zip(abs_paths, rel_paths), total=len(abs_paths)):
            with Image.open(abs_path) as img:
                orig_w, orig_h = img.size
                image_np = np.array(img.convert("RGB"))

            # Pass a dummy keypoint to satisfy Albumentations validation requirements
            transformed = val_tf(image=image_np, keypoints=[(0, 0)])
            image_tensor = transformed["image"].unsqueeze(0).to(device)

            coord_pred, class_logits = model(image_tensor)

            pred_x_norm = coord_pred[0, 0].item()
            pred_y_norm = coord_pred[0, 1].item()

            pred_x_orig = pred_x_norm * orig_w
            pred_y_orig = pred_y_norm * orig_h

            pred_class_idx = torch.argmax(class_logits, dim=1).item()
            pred_shape_name = SHAPE_CLASSES[pred_class_idx]

            predictions[rel_path] = {
                "mark": {
                    "x": float(np.round(pred_x_orig, 2)),
                    "y": float(np.round(pred_y_orig, 2))
                },
                "verified_shape": pred_shape_name
            }

    print(f"Saving predictions to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f, indent=4)

    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
