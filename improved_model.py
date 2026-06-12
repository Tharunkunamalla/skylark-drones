import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    """
    U-Net Decoder block: Upsample -> Concat skip connections -> Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Conv to merge skip connection and input
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        # Bilinear upsampling by 2x
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class GCPUNetMultitaskModel(nn.Module):
    """
    Improved multitask model utilizing U-Net for keypoint heatmap prediction
    and a classification head at the bottleneck.
    
    Backbone Encoder: ResNet18
    Outputs:
        - heatmap: (batch_size, 1, H, W) 2D Gaussian heatmap in range [0, 1]
        - shape_logits: (batch_size, 3) classification logits for shape classes
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Load encoder backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Encoder skip layers
        self.enc_conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        ) # Output: (64, H/2, W/2)
        
        self.enc_maxpool = backbone.maxpool
        self.enc_layer1 = backbone.layer1 # Output: (64, H/4, W/4)
        self.enc_layer2 = backbone.layer2 # Output: (128, H/8, W/8)
        self.enc_layer3 = backbone.layer3 # Output: (256, H/16, W/16)
        self.enc_layer4 = backbone.layer4 # Output: (512, H/32, W/32)
        
        # Decoder blocks
        # Block 3: input layer4, skip layer3
        self.dec_block3 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # Block 2: input Block 3 output, skip layer2
        self.dec_block2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # Block 1: input Block 2 output, skip layer1
        self.dec_block1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # Block 0: input Block 1 output, skip conv1
        self.dec_block0 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        
        # Final head to produce full-resolution heatmap (H, W)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Restricts values to [0, 1] range for heatmap probabilities
        )
        
        # Classification Head at the encoder bottleneck (layer4 output)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # --- Encoder forward ---
        s0 = self.enc_conv1(x)                # [B, 64, H/2, W/2]
        x_pool = self.enc_maxpool(s0)         # [B, 64, H/4, W/4]
        s1 = self.enc_layer1(x_pool)          # [B, 64, H/4, W/4]
        s2 = self.enc_layer2(s1)              # [B, 128, H/8, W/8]
        s3 = self.enc_layer3(s2)              # [B, 256, H/16, W/16]
        s4 = self.enc_layer4(s3)              # [B, 512, H/32, W/32] (bottleneck)
        
        # --- Classification Head forward ---
        shape_logits = self.classification_head(s4) # [B, num_classes]
        
        # --- Decoder forward ---
        d3 = self.dec_block3(s4, s3)          # [B, 256, H/16, W/16]
        d2 = self.dec_block2(d3, s2)          # [B, 128, H/8, W/8]
        d1 = self.dec_block1(d2, s1)          # [B, 64, H/4, W/4]
        d0 = self.dec_block0(d1, s0)          # [B, 32, H/2, W/2]
        
        # Final upsampling to original input shape [B, 1, H, W]
        d_up = nn.functional.interpolate(d0, scale_factor=2, mode="bilinear", align_corners=True) # [B, 32, H, W]
        features_up = self.final_upsample(d_up)  # [B, 16, H, W]
        heatmap = self.heatmap_head(features_up) # [B, 1, H, W]
        
        return heatmap, shape_logits

if __name__ == "__main__":
    print("Initializing UNet multitask model...")
    model = GCPUNetMultitaskModel(pretrained=False)
    
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    heatmap, logits = model(dummy_input)
    print("\nVerifying network output shape:")
    print(f"  - Heatmap output shape:        {heatmap.shape} (Expected: [2, 1, 224, 224])")
    print(f"  - Classification output shape: {logits.shape} (Expected: [2, 3])")
    
    # Check output ranges
    print(f"  - Heatmap values range:        min={heatmap.min().item():.3f}, max={heatmap.max().item():.3f} (Expected range: [0, 1])")
    
    assert heatmap.shape == (2, 1, 224, 224), "Incorrect heatmap output shape!"
    assert logits.shape == (2, 3), "Incorrect classification output shape!"
    print("\nUNet Model sanity check passed successfully!")
