"""
Papal Charter Classification Training Pipeline

Trains a CNN-based classifier to distinguish papal charters from non-papal charters
using transfer learning with EfficientNet-B0.

Dataset structure expected:
data/
  train/
    papal/
    non_papal/
  val/
    papal/
    non_papal/
  test/
    papal/
    non_papal/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CharterDataset(Dataset):
    """custom dataset for charter images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['non_papal', 'papal', 'papal_canapis', 'non_papal_solemn']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} images in {root_dir}")
        for class_name in self.classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            print(f"  - {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224, augment=True):
    """Define image transformations"""
    
    if augment:
        # training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.3),  # May not be appropriate for all charters
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # basic training transform without augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes=4, pretrained=True):
    """Create EfficientNet-B0 model with custom classifier"""
    
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # freeze early layers (unfreeze later for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
    
    # replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(dataloader), 'acc': 100.*correct/total})
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of papal class
    
    return (running_loss / len(dataloader), 
            100. * correct / total, 
            np.array(all_preds), 
            np.array(all_labels),
            np.array(all_probs))


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """Plot ROC curve"""
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def main():
    # Configuration
    config = {
        'data_dir': 'data_split',
        'batch_size': 16,
        'image_size': 224,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'num_classes': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'results',
        'early_stopping_patience': 7
    }
    
    print("=" * 60)
    print("Papal Charter Classification Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['image_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=config['image_size'], 
        augment=True
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CharterDataset(
        os.path.join(config['data_dir'], 'train'),
        transform=train_transform
    )
    val_dataset = CharterDataset(
        os.path.join(config['data_dir'], 'val'),
        transform=val_transform
    )
    test_dataset = CharterDataset(
        os.path.join(config['data_dir'], 'test'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=2)
    
    # Create model
    print("\nInitializing model...")
    model = create_model(num_classes=config['num_classes'])
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    # Phase 1: Train only the classifier
    print("\nPhase 1: Training classifier head only")
    for epoch in range(config['num_epochs'] // 2):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']//2}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, config['device'])
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Phase 2: Fine-tune entire network
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning entire network")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'] / 10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    
    patience_counter = 0
    
    for epoch in range(config['num_epochs'] // 2):
        print(f"\nEpoch {len(history['train_loss'])+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, config['device'])
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {len(history['train_loss'])} total epochs")
            break
    
    # Plot training history
    print("\n" + "=" * 60)
    print("Generating training visualizations...")
    plot_training_history(history, os.path.join(config['save_dir'], 'training_history.png'))
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'best_model.pth')))
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, config['device']
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    classes = ['Non-Papal', 'Papal', 'Papal Canapis', 'Non-Papal Solemn']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, 
                         classes=classes,
                         save_path=os.path.join(config['save_dir'], 'confusion_matrix.png'))
    
    # Plot ROC curve
    #plot_roc_curve(test_labels, test_probs,
    #              save_path=os.path.join(config['save_dir'], 'roc_curve.png'))
    
    # Save final metrics
    final_metrics = {
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'classification_report': classification_report(test_labels, test_preds, 
                                                       target_names=classes,
                                                       output_dict=True)
    }
    
    with open(os.path.join(config['save_dir'], 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Results saved to: {config['save_dir']}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
