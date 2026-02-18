"""
Papal Charter Classification - Inference Script

Use this script to classify new charter images as papal or non-papal
using a trained model.

Usage:
    python predict.py --image path/to/charter.jpg
    python predict.py --directory path/to/charters/
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path
import json


class CharterClassifier:
    """Wrapper class for charter classification"""
    
    def __init__(self, model_path='results/best_model.pth', device=None):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model on device: {self.device}")
        
        # Create model architecture
        self.model = models.efficientnet_b0(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 2)
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.classes = ['Non-Papal', 'Papal']
        
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
                'Non-Papal': probs[0, 0].item(),
                'Papal': probs[0, 1].item()
            }
        
        return result
    
    def predict_directory(self, directory_path, output_file=None):
        """
        Classify all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to save results as JSON
        
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
        
        # Print summary
        papal_count = sum(1 for r in results if r['prediction'] == 'Papal')
        non_papal_count = sum(1 for r in results if r['prediction'] == 'Non-Papal')
        
        print("\n" + "=" * 60)
        print("Classification Summary:")
        print(f"  Total images: {len(results)}")
        print(f"  Papal: {papal_count}")
        print(f"  Non-Papal: {non_papal_count}")
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
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results (only for --directory mode)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to run inference on (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not args.image and not args.directory:
        parser.error("Must specify either --image or --directory")
    
    # Initialize classifier
    classifier = CharterClassifier(model_path=args.model, device=args.device)
    
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
        
        results = classifier.predict_directory(args.directory, args.output)


if __name__ == '__main__':
    main()
