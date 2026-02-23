"""
Visualization and Analysis Tools for Papal Charter Classifier

This script provides tools to:
1. Visualize model predictions with GradCAM
2. Analyze misclassifications
3. Compare predictions across Austrian vs French charters

Usage:
    # Visualize a single prediction with attention map
    python visualize.py --mode single --image path/to/charter.jpg
    
    # Analyze test set performance
    python visualize.py --mode analyze
    
    # Generate GradCAM visualizations for misclassifications
    python visualize.py --mode misclassifications
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import sys

# Import CharterClassifier from predict.py
sys.path.insert(0, str(Path(__file__).parent))
from predict import CharterClassifier


class GradCAM:
    """Gradient-weighted Class Activation Mapping for visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate GradCAM heatmap"""
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Compute weights
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def visualize_prediction(classifier, image_path, save_path=None):
    """
    Visualize a prediction with GradCAM heatmap
    
    Args:
        classifier: CharterClassifier instance
        image_path: Path to image
        save_path: Optional path to save visualization
    """
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    input_tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
    
    # Get prediction
    with torch.no_grad():
        outputs = classifier.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    # Get GradCAM
    target_layer = classifier.model.features[-1]
    gradcam = GradCAM(classifier.model, target_layer)
    heatmap = gradcam.generate(input_tensor, target_class=predicted.item())
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    
    # Create overlay
    img_array = np.array(image)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Attention Map (GradCAM)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {classifier.classes[predicted.item()]}\n'
                      f'Confidence: {confidence.item():.3f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'image': str(image_path),
        'prediction': classifier.classes[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': {
            classifier.classes[i]: probs[0, i].item() 
            for i in range(len(classifier.classes))
        }
    }


def analyze_test_results(results_dir='results', data_dir='data/test'):
    """
    Analyze test set predictions and identify misclassifications
    
    Args:
        results_dir: Directory containing model and results
        data_dir: Directory containing test images
    """
    
    # Load model
    from predict import CharterClassifier
    classifier = CharterClassifier(model_path=f'{results_dir}/best_model.pth')
    
    data_path = Path(data_dir)
    
    results = {
        'papal': {'correct': [], 'incorrect': []},
        'non_papal': {'correct': [], 'incorrect': []}
    }
    
    print("Analyzing test set predictions...")
    print("=" * 60)
    
    # Analyze papal charters
    papal_dir = data_path / 'papal'
    if papal_dir.exists():
        for img_path in papal_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                pred = classifier.predict_single(img_path, return_prob=True)
                
                if pred['prediction'] == 'Papal':
                    results['papal']['correct'].append(pred)
                else:
                    results['papal']['incorrect'].append(pred)
    
    # Analyze non-papal charters
    non_papal_dir = data_path / 'non_papal'
    if non_papal_dir.exists():
        for img_path in non_papal_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                pred = classifier.predict_single(img_path, return_prob=True)
                
                if pred['prediction'] == 'Non-Papal':
                    results['non_papal']['correct'].append(pred)
                else:
                    results['non_papal']['incorrect'].append(pred)
    
    # Print summary
    papal_correct = len(results['papal']['correct'])
    papal_incorrect = len(results['papal']['incorrect'])
    non_papal_correct = len(results['non_papal']['correct'])
    non_papal_incorrect = len(results['non_papal']['incorrect'])
    
    total_correct = papal_correct + non_papal_correct
    total_incorrect = papal_incorrect + non_papal_incorrect
    total = total_correct + total_incorrect
    
    print(f"\nTest Set Analysis:")
    print(f"Total images: {total}")
    print(f"Correct: {total_correct} ({100*total_correct/total:.1f}%)")
    print(f"Incorrect: {total_incorrect} ({100*total_incorrect/total:.1f}%)")
    print()
    print(f"Papal charters:")
    print(f"  Correct: {papal_correct}")
    print(f"  Incorrect: {papal_incorrect} (misclassified as Non-Papal)")
    print()
    print(f"Non-Papal charters:")
    print(f"  Correct: {non_papal_correct}")
    print(f"  Incorrect: {non_papal_incorrect} (misclassified as Papal)")
    
    # Identify most confident mistakes
    if papal_incorrect > 0:
        print("\nMost confident papal → non-papal mistakes:")
        sorted_mistakes = sorted(results['papal']['incorrect'], 
                                key=lambda x: x['confidence'], reverse=True)
        for i, mistake in enumerate(sorted_mistakes[:3], 1):
            print(f"  {i}. {Path(mistake['image']).name} "
                  f"(confidence: {mistake['confidence']:.3f})")
    
    if non_papal_incorrect > 0:
        print("\nMost confident non-papal → papal mistakes:")
        sorted_mistakes = sorted(results['non_papal']['incorrect'], 
                                key=lambda x: x['confidence'], reverse=True)
        for i, mistake in enumerate(sorted_mistakes[:3], 1):
            print(f"  {i}. {Path(mistake['image']).name} "
                  f"(confidence: {mistake['confidence']:.3f})")
    
    print("=" * 60)
    
    # Save detailed results
    output_file = f'{results_dir}/test_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")
    
    return results


def visualize_misclassifications(results_dir='results', data_dir='data/test', 
                                 max_examples=5):
    """
    Create GradCAM visualizations for misclassified examples
    
    Args:
        results_dir: Directory containing model
        data_dir: Directory containing test images
        max_examples: Maximum number of misclassifications to visualize per class
    """
    
    from predict import CharterClassifier
    
    # Analyze test set
    results = analyze_test_results(results_dir, data_dir)
    
    # Load classifier
    classifier = CharterClassifier(model_path=f'{results_dir}/best_model.pth')
    
    # Create output directory
    vis_dir = Path(results_dir) / 'misclassifications'
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations for misclassifications...")
    
    # Visualize papal misclassifications
    if results['papal']['incorrect']:
        print(f"\nVisualizing papal → non-papal mistakes...")
        for i, mistake in enumerate(results['papal']['incorrect'][:max_examples], 1):
            save_path = vis_dir / f'papal_mistake_{i}.png'
            visualize_prediction(classifier, mistake['image'], save_path)
    
    # Visualize non-papal misclassifications
    if results['non_papal']['incorrect']:
        print(f"\nVisualizing non-papal → papal mistakes...")
        for i, mistake in enumerate(results['non_papal']['incorrect'][:max_examples], 1):
            save_path = vis_dir / f'non_papal_mistake_{i}.png'
            visualize_prediction(classifier, mistake['image'], save_path)
    
    print(f"\nVisualizations saved to {vis_dir}/")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize and analyze papal charter classifier'
    )
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'analyze', 'misclassifications'],
                       help='Visualization mode')
    parser.add_argument('--image', type=str,
                       help='Image path (for single mode)')
    parser.add_argument('--output', type=str,
                       help='Output path to save visualization (for single mode)')
    parser.add_argument('--model', type=str, default='results/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--test-dir', type=str, default='data/test',
                       help='Test data directory')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.image:
            parser.error("--image required for single mode")
        
        classifier = CharterClassifier(model_path=args.model)
        result = visualize_prediction(classifier, args.image, save_path=args.output)
        
        # Print results to console
        print(f"\nImage: {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")
    
    elif args.mode == 'analyze':
        analyze_test_results(args.results_dir, args.test_dir)
    
    elif args.mode == 'misclassifications':
        visualize_misclassifications(args.results_dir, args.test_dir)


if __name__ == '__main__':
    main()
