# Aerial GCP Pose Estimation and Shape Classification

This repository contains a complete machine learning pipeline built to automate the identification of Ground Control Points (GCPs) in aerial surveying imagery. The system solves two problems simultaneously (multitask learning):
1. **Keypoint Localization (Pose Estimation)**: Predict the exact `(x, y)` pixel coordinates of the center of the GCP marker.
2. **Shape Classification**: Classify the physical shape of the marker into one of three classes: `Cross`, `Square`, or `L-Shaped`.

---

## 1. Project Directory Structure

```
.
├── train_dataset/                  # Curated training data images (omitted from repository tracking)
├── test_dataset/                   # Raw testing data images (omitted from repository tracking)
│
├── eda.ipynb                       # Exploratory Data Analysis notebook (Phase 1)
├── error_analysis.ipynb            # Error analysis and visual audit notebook (Phase 9)
├── class_distribution.png          # Shape class distribution chart
├── sample_visualization.png        # Random dataset samples with overlaid ground truth
├── error_analysis_visualization.png# Best vs worst localization predictions crop visualization
│
├── dataset.py                      # Custom PyTorch Dataset & stratified validation split loader (Phase 2)
├── transforms.py                   # Albumentations preprocessing & keypoint scaling transforms (Phase 3)
│
├── model.py                        # Baseline Multitask ResNet18 model architecture (Phase 4)
├── improved_model.py               # Improved Multitask ResNet18-UNet model architecture (Phase 7)
├── train.py                        # Multitask model training pipeline (Phase 5)
├── evaluate.py                     # Metric calculation on validation set (Phase 6)
├── infer.py                        # Batch inference generator on test dataset (Phase 8)
│
├── requirements.txt                # Pinned package requirements
├── evaluation_report.txt           # Printout of the baseline model metrics
├── predictions.json                # Generated predictions on the unlabelled test dataset
└── README.md                       # Documentation (Phase 10)
```

---

## 2. Approach

1. **Physical File Mismatch Handling**: The annotations JSON contains 1,000 mappings, but the ZIP archive physically contains only **613** images. The dataset parser dynamically scans the disk, filtering out the 387 missing files.
2. **Missing Label Handling**: 2 of the 613 physically existing images lack a shape classification label. These are excluded from splits to ensure correct shape classification inputs.
3. **Class Imbalance Mitigation**: The 611 valid images are split into training (488) and validation (123) sets using a **stratified split** to preserve the proportion of the under-represented `Cross` class (which has only 58 samples overall, approx. 9.5%).
4. **Coordinate Normalization**: To prevent exploding gradients and ensure model convergence, target coordinates `(x, y)` are normalized to `[0, 1]` in the model and loss calculations, and mapped back to original image dimensions for evaluation and predictions.

---

## 3. Model Architectures

### A. Baseline Model (`model.py`)
- **Backbone**: Pre-trained ResNet18 (ImageNet).
- **Classification Head**: MLP branching from collapsed bottleneck features, outputting 3 class logits.
- **Coordinate Head**: MLP branching from collapsed bottleneck features, outputting `(x, y)` in `[0, 1]` via Sigmoid activation.
- **Limitation**: Global average pooling in ResNet18 collapses spatial dimensions from `7x7` to `1x1`, losing precise keypoint coordinates. Consequently, the baseline coordinate head converges to the center prediction `(0.5, 0.5)` for all inputs.

### B. Improved Model (`improved_model.py`)
- **Heatmap-based U-Net**: A encoder-decoder architecture utilizing skip connections.
- **Decoder Blocks**: 4 blocks upsampling bottleneck features and merging skip connections from intermediate ResNet layers (`conv1`, `layer1`, `layer2`, `layer3`) to preserve high-resolution spatial feature maps.
- **Heatmap Head**: Outputs a 2D Gaussian probability distribution heatmap of shape `[batch_size, 1, H, W]`. Coordinate extraction is achieved via argmax.
- **Classification Head**: Connected to the encoder bottleneck (`layer4`) using global average pooling and a fully connected layer.

---

## 4. Training Strategy

- **Loss Optimization**:
  - Keypoint Regression: Mean Squared Error (MSE) loss on normalized `[0, 1]` coordinates.
  - Shape Classification: CrossEntropyLoss.
  - Multi-task combination: `Loss = 10.0 * MSE_Loss + 1.0 * CrossEntropy_Loss`.
- **Optimization Settings**: Adam optimizer, learning rate `3e-4`.
- **Augmentations**: Keypoint-aware image resizing (via Albumentations `Resize` and `KeypointParams`), ImageNet normalization.

---

## 5. Assumptions

1. **Coordinate Alignment**: Image coordinates are mapped relative to the top-left corner `(0, 0)`.
2. **Label Discrepancy**: The training JSON lists `"L-Shape"`, but requirements demand the output class mapping to be `"L-Shaped"`. We map class index 2 back to `"L-Shaped"` in `predictions.json`.
3. **Evaluation Resolution**: Evaluation metrics (Mean Pixel Error, PCK) are calculated in the original high-resolution coordinate space (`4096x2730` or `4096x3068`) by scaling predictions by the original image dimensions.

---

## 6. How to Run & Reproduce

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Model Training
To train the baseline model:
```bash
python train.py --epochs 15 --batch_size 16 --img_size 224
```
This saves the best checkpoint to `best_model.pth`.

### Step 3: Run Evaluation
To evaluate the baseline checkpoint on the validation split and output metrics:
```bash
python evaluate.py
```
This prints the metrics and saves the results to `evaluation_report.txt`.

### Step 4: Run Inference
To run batch inference on the unlabelled test dataset:
```bash
python infer.py
```
This generates `predictions.json` in the project root containing the coordinates and shape predictions.
