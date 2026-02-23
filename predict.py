"""
Papal Charter Classification - Inference Script

Use this script to classify charter images using a trained model.
Supports both binary (2-class) and multi-class models.

Usage:
    # Classify a single image
    python predict.py --image path/to/charter.jpg
    
    # Classify all images in a directory
    python predict.py --directory path/to/charters/
    
    # Classify and save results to JSON
    python predict.py --directory path/to/charters/ --output results.json
    
    # For BINARY models (2 classes):
    # Copy papal images to a separate directory
    python predict.py --directory path/to/charters/ --copy-to papal_charters/
    
    # For MULTI-CLASS models (4 classes):
    # Auto-copies all classes containing 'Papal' (both Simple and Solemn)
    python predict.py --directory path/to/charters/ --copy-to papal_charters/
    
    # Copy only specific classes
    python predict.py --directory path/to/charters/ --copy-to solemn_papal/ --copy-classes "Solemn Papal"
    
    # Copy multiple specific classes
    python predict.py --directory path/to/charters/ --copy-to papal_all/ --copy-classes "Simple Papal" "Solemn Papal"
    
    # Copy only high-confidence predictions (e.g., >= 0.8)
    python predict.py --directory path/to/charters/ --copy-to papal_charters/ --min-confidence 0.8
    
    # Override default class names (must match model's number of classes)
    python predict.py --directory path/to/charters/ --classes "Other" "Simple Papal" "Solemn Non-Papal" "Solemn Papal"
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path
import json
import shutil


class CharterClassifier:
    """Wrapper class for charter classification"""
    
    def __init__(self, model_path='results/best_model.pth', device=None, class_names=None):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
            class_names: List of class names (optional, will auto-detect from model if not provided)
        """
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model on device: {self.device}")
        
        # Load checkpoint to determine number of classes
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Infer number of classes from checkpoint
        if 'classifier.1.weight' in checkpoint:
            num_classes = checkpoint['classifier.1.weight'].shape[0]
        elif 'classifier.1.bias' in checkpoint:
            num_classes = checkpoint['classifier.1.bias'].shape[0]
        else:
            raise ValueError("Cannot determine number of classes from checkpoint")
        
        print(f"Detected {num_classes}-class model")
        
        # Create model architecture
        self.model = models.efficientnet_b0(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Set class names
        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(f"Provided {len(class_names)} class names but model has {num_classes} classes")
            self.classes = class_names
        else:
            # Default class names based on number of classes
            if num_classes == 2:
                self.classes = ['non_papal', 'papal']
            elif num_classes == 4:
                self.classes = ['non_papal', 'papal', 'papal_canapis', 'non_papal_solemn']
            else:
                self.classes = [f'Class_{i}' for i in range(num_classes)]
        
        print(f"Class names: {self.classes}")
        print("Model loaded successfully!")
    
    def predict_single(self, image_path, return_prob=True):
        """
        Classify a single image
        
        Args:
            image_path: Path to image file
            return_prob: Whether to return probabilities
        
        Returns:
            dict with prediction, confidence, and optionally probabilities
        """
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        result = {
            'image': str(image_path),
            'prediction': self.classes[predicted.item()],
            'confidence': confidence.item()
        }
        
        if return_prob:
            result['probabilities'] = {
                self.classes[i]: probs[0, i].item() 
                for i in range(len(self.classes))
            }
        
        return result
    
    def predict_directory(self, directory_path, output_file=None, copy_to=None, copy_classes=None, min_confidence=None):
        """
        Classify all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to save results as JSON
            copy_to: Optional path to directory where selected images should be copied
            copy_classes: List of class names to copy (default: all classes containing 'Papal')
            min_confidence: Optional minimum confidence threshold for copying (default: None, copies all)
        
        Returns:
            list of prediction dictionaries
        """
        
        directory = Path(directory_path)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(list(directory.glob(ext)))
            image_files.extend(list(directory.glob(ext.upper())))
        
        if not image_files:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"Found {len(image_files)} images to classify")
        
        results = []
        for img_path in image_files:
            try:
                result = self.predict_single(img_path)
                results.append(result)
                print(f"✓ {img_path.name}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.3f})")
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        # Copy selected images if directory specified
        if copy_to:
            copy_path = Path(copy_to)
            copy_path.mkdir(parents=True, exist_ok=True)
            
            # Determine which classes to copy
            if copy_classes is None:
                # Default: copy all classes containing 'Papal' (case-insensitive)
                copy_classes = [cls for cls in self.classes if 'papal' in cls.lower()]
                print(f"\nAuto-detected classes to copy: {copy_classes}")
            else:
                # Validate provided class names
                invalid_classes = [cls for cls in copy_classes if cls not in self.classes]
                if invalid_classes:
                    print(f"Warning: Invalid class names will be ignored: {invalid_classes}")
                    copy_classes = [cls for cls in copy_classes if cls in self.classes]
            
            if not copy_classes:
                print("Warning: No valid classes specified for copying. Skipping copy operation.")
            else:
                # Filter results by selected classes
                selected_results = [r for r in results if r['prediction'] in copy_classes]
                
                # Apply confidence filter if specified
                if min_confidence is not None:
                    selected_results = [r for r in selected_results if r['confidence'] >= min_confidence]
                
                print(f"\nCopying {len(selected_results)} images (classes: {copy_classes}) to {copy_to}...")
                
                copied_count = 0
                for result in selected_results:
                    try:
                        src = Path(result['image'])
                        dst = copy_path / src.name
                        
                        # Handle name collisions by adding suffix
                        if dst.exists():
                            stem = dst.stem
                            suffix = dst.suffix
                            counter = 1
                            while dst.exists():
                                dst = copy_path / f"{stem}_{counter}{suffix}"
                                counter += 1
                        
                        shutil.copy2(src, dst)
                        copied_count += 1
                        conf_str = f" (class: {result['prediction']}, confidence: {result['confidence']:.3f})"
                        if min_confidence is not None:
                            conf_str += f" [threshold: {min_confidence:.3f}]"
                        print(f"  ✓ Copied {src.name}{conf_str}")
                    except Exception as e:
                        print(f"  ✗ Error copying {src.name}: {e}")
                
                print(f"\nSuccessfully copied {copied_count} images")
                if min_confidence is not None:
                    skipped = len([r for r in results if r['prediction'] in copy_classes]) - copied_count
                    if skipped > 0:
                        print(f"Skipped {skipped} images below confidence threshold {min_confidence:.3f}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Classification Summary:")
        print(f"  Total images: {len(results)}")
        
        # Count by class
        for class_name in self.classes:
            count = sum(1 for r in results if r['prediction'] == class_name)
            print(f"  {class_name}: {count}")
        
        print("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Classify charter images as papal or non-papal'
    )
    parser.add_argument('--image', type=str,
                       help='Path to a single image to classify')
    parser.add_argument('--directory', type=str,
                       help='Path to directory containing images to classify')
    parser.add_argument('--model', type=str, default='results/best_model.pth',
                       help='Path to trained model (default: results/best_model.pth)')
    parser.add_argument('--classes', type=str, nargs='+',
                       help='Custom class names in order (e.g., --classes "Other" "Simple Papal" "Solemn Non-Papal" "Solemn Papal")')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results (only for --directory mode)')
    parser.add_argument('--copy-to', type=str,
                       help='Directory to copy selected images to (only for --directory mode)')
    parser.add_argument('--copy-classes', type=str, nargs='+',
                       help='Class names to copy (default: all classes containing "Papal")')
    parser.add_argument('--min-confidence', type=float,
                       help='Minimum confidence threshold for copying (0.0-1.0, default: copy all)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to run inference on (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not args.image and not args.directory:
        parser.error("Must specify either --image or --directory")
    
    # Initialize classifier
    classifier = CharterClassifier(
        model_path=args.model, 
        device=args.device,
        class_names=args.classes
    )
    
    # Single image mode
    if args.image:
        print("\n" + "=" * 60)
        print("Classifying single image...")
        print("=" * 60)
        
        result = classifier.predict_single(args.image)
        
        print(f"\nImage: {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
    
    # Directory mode
    elif args.directory:
        print("\n" + "=" * 60)
        print("Classifying images in directory...")
        print("=" * 60)
        
        # Validate arguments
        if args.min_confidence is not None and not args.copy_to:
            parser.error("--min-confidence requires --copy-to to be specified")
        
        if args.copy_classes is not None and not args.copy_to:
            parser.error("--copy-classes requires --copy-to to be specified")
        
        if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
            parser.error("--min-confidence must be between 0.0 and 1.0")
        
        results = classifier.predict_directory(
            args.directory, 
            args.output, 
            copy_to=args.copy_to,
            copy_classes=args.copy_classes,
            min_confidence=args.min_confidence
        )


if __name__ == '__main__':
    main()
