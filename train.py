import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from dataset import get_dataloaders
from transforms import get_train_transforms, get_val_transforms
from model import GCPMultitaskModel

def train_one_epoch(model, dataloader, optimizer, criterion_coord, criterion_class, device, img_size, w_coord, w_class):
    model.train()
    running_loss = 0.0
    running_loss_coord = 0.0
    running_loss_class = 0.0
    
    for batch in dataloader:
        images = batch["image"].to(device)
        shape_targets = batch["shape_class"].to(device)
        
        x_norm = batch["x"] / img_size
        y_norm = batch["y"] / img_size
        coord_targets = torch.stack([x_norm, y_norm], dim=1).to(device)
        
        optimizer.zero_grad()
        
        coord_preds, class_logits = model(images)
        
        loss_coord = criterion_coord(coord_preds, coord_targets)
        loss_class = criterion_class(class_logits, shape_targets)
        loss = w_coord * loss_coord + w_class * loss_class
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_loss_coord += loss_coord.item() * images.size(0)
        running_loss_class += loss_class.item() * images.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_loss_coord = running_loss_coord / len(dataloader.dataset)
    epoch_loss_class = running_loss_class / len(dataloader.dataset)
    
    return epoch_loss, epoch_loss_coord, epoch_loss_class

def validate(model, dataloader, criterion_coord, criterion_class, device, img_size, w_coord, w_class):
    model.eval()
    running_loss = 0.0
    running_loss_coord = 0.0
    running_loss_class = 0.0
    
    all_class_preds = []
    all_class_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            shape_targets = batch["shape_class"].to(device)
            
            x_norm = batch["x"] / img_size
            y_norm = batch["y"] / img_size
            coord_targets = torch.stack([x_norm, y_norm], dim=1).to(device)
            
            coord_preds, class_logits = model(images)
            
            loss_coord = criterion_coord(coord_preds, coord_targets)
            loss_class = criterion_class(class_logits, shape_targets)
            loss = w_coord * loss_coord + w_class * loss_class
            
            running_loss += loss.item() * images.size(0)
            running_loss_coord += loss_coord.item() * images.size(0)
            running_loss_class += loss_class.item() * images.size(0)
            
            class_preds = torch.argmax(class_logits, dim=1)
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_targets.extend(shape_targets.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    val_loss_coord = running_loss_coord / len(dataloader.dataset)
    val_loss_class = running_loss_class / len(dataloader.dataset)
    
    accuracy = accuracy_score(all_class_targets, all_class_preds)
    macro_f1 = f1_score(all_class_targets, all_class_preds, average="macro")
    
    return val_loss, val_loss_coord, val_loss_class, accuracy, macro_f1

def main():
    parser = argparse.ArgumentParser(description="GCP Multitask Baseline Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=224, help="Input size to resize images to")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--w_coord", type=float, default=10.0, help="Weight for coordinate loss")
    parser.add_argument("--w_class", type=float, default=1.0, help="Weight for classification loss")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save best model weights")
    args = parser.parse_args()

    JSON_PATH = os.path.join("train_dataset", "train_dataset", "gcp_marks.json")
    BASE_DIR = os.path.join("train_dataset", "train_dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf = get_train_transforms(img_size=args.img_size)
    val_tf = get_val_transforms(img_size=args.img_size)

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        BASE_DIR,
        JSON_PATH,
        batch_size=args.batch_size,
        val_split=0.2,
        train_transform=train_tf,
        val_transform=val_tf
    )

    print("Initializing multitask ResNet18 model...")
    model = GCPMultitaskModel(num_classes=3, pretrained=True).to(device)

    criterion_coord = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    
    print("\nStarting Training Pipeline:")
    for epoch in range(args.epochs):
        train_loss, train_loss_coord, train_loss_class = train_one_epoch(
            model, train_loader, optimizer, criterion_coord, criterion_class, device, args.img_size, args.w_coord, args.w_class
        )
        
        val_loss, val_loss_coord, val_loss_class, val_acc, val_f1 = validate(
            model, val_loader, criterion_coord, criterion_class, device, args.img_size, args.w_coord, args.w_class
        )
        
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} (Coord MSE: {train_loss_coord:.5f}, Class CE: {train_loss_class:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Coord MSE: {val_loss_coord:.5f}, Class CE: {val_loss_class:.4f}) | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  => Saved best model to {args.save_path} (Val Loss: {val_loss:.4f})")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
