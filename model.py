import torch
import torch.nn as nn
import torchvision.models as models

class GCPMultitaskModel(nn.Module):
    """
    Multitask model for GCP center localization (regression)
    and physical marker shape classification.
    
    Backbone: ResNet18 (pretrained on ImageNet)
    Outputs:
        - coords: (batch_size, 2) normalized [x, y] coordinates in range [0, 1]
        - shape_logits: (batch_size, 3) classification logits for [Cross, Square, L-Shape]
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Load backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Get feature dimensions
        in_features = self.backbone.fc.in_features
        
        # Replace the original classification head with Identity
        self.backbone.fc = nn.Identity()
        
        # Branch 1: Coordinate Regression Head (predicts x and y)
        self.coordinate_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()  # Restricts keypoint coordinates to [0, 1] range
        )
        
        # Branch 2: Shape Classification Head (predicts shape class)
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Get head outputs
        coords = self.coordinate_head(features)
        shape_logits = self.classification_head(features)
        
        return coords, shape_logits

if __name__ == "__main__":
    # Quick sanity check with dummy input
    print("Initializing model...")
    model = GCPMultitaskModel(pretrained=False)
    
    dummy_input = torch.randn(4, 3, 512, 512)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    coords, logits = model(dummy_input)
    print("\nVerifying network output shape:")
    print(f"  - Coordinate output shape:     {coords.shape} (Expected: [4, 2])")
    print(f"  - Classification output shape: {logits.shape} (Expected: [4, 3])")
    
    # Check output ranges
    print(f"  - Coordinate values range:     min={coords.min().item():.3f}, max={coords.max().item():.3f} (Expected range: [0, 1])")
    
    assert coords.shape == (4, 2), "Incorrect coordinate output shape!"
    assert logits.shape == (4, 3), "Incorrect classification output shape!"
    print("\nModel sanity check passed successfully!")
