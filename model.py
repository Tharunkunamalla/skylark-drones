import torch
import torch.nn as nn
import torchvision.models as models

class GCPMultitaskModel(nn.Module):
    """
    Multitask ResNet-based model for ground control point (GCP) center localization 
    and shape classification.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.coordinate_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        coords = self.coordinate_head(features)
        shape_logits = self.classification_head(features)
        return coords, shape_logits

if __name__ == "__main__":
    print("Initializing model...")
    model = GCPMultitaskModel(pretrained=False)
    
    dummy_input = torch.randn(4, 3, 512, 512)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    coords, logits = model(dummy_input)
    print("\nVerifying network output shape:")
    print(f"  - Coordinate output shape:     {coords.shape} (Expected: [4, 2])")
    print(f"  - Classification output shape: {logits.shape} (Expected: [4, 3])")
    
    print(f"  - Coordinate values range:     min={coords.min().item():.3f}, max={coords.max().item():.3f}")
    
    assert coords.shape == (4, 2), "Incorrect coordinate output shape!"
    assert logits.shape == (4, 3), "Incorrect classification output shape!"
    print("\nModel sanity check passed successfully!")
