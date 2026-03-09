"""
Batch GradCAM Visualization Script

This script generates GradCAM visualizations for multiple images at once,
useful for systematically analyzing what your model learned.

Usage:
    # Visualize all images in a directory
    python batch_visualize.py --directory path/to/charters/ --output vis_output/
    
    # Visualize with specific class filter
    python batch_visualize.py --directory path/to/charters/ --output vis_output/ --filter-class "Solemn Papal"
    
    # Limit number of images
    python batch_visualize.py --directory path/to/charters/ --output vis_output/ --max-images 10
"""

import argparse
from pathlib import Path
from predict import CharterClassifier
from visualize import visualize_prediction


def batch_visualize(image_dir, output_dir, model_path='results/best_model.pth', 
                   filter_class=None, max_images=None):
    """
    Generate GradCAM visualizations for multiple images
    
    Args:
        image_dir: Directory containing images to visualize
        output_dir: Directory to save visualizations
        model_path: Path to trained model
        filter_class: Only visualize images predicted as this class (optional)
        max_images: Maximum number of images to process (optional)
    """
    
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(list(image_path.glob(ext)))
        image_files.extend(list(image_path.glob(ext.upper())))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Load classifier
    print("Loading model...")
    classifier = CharterClassifier(model_path=model_path)
    
    if filter_class and filter_class not in classifier.classes:
        print(f"Warning: '{filter_class}' is not a valid class name")
        print(f"Valid classes: {classifier.classes}")
        return
    
    # Process images
    processed = 0
    filtered = 0
    
    print(f"\nGenerating visualizations...")
    print("=" * 60)
    
    for img_file in image_files:
        if max_images and processed >= max_images:
            print(f"\nReached maximum of {max_images} images")
            break
        
        try:
            # Generate visualization
            output_file = output_path / f"vis_{img_file.stem}.png"
            result = visualize_prediction(classifier, img_file, save_path=str(output_file))
            
            # Check class filter
            if filter_class and result['prediction'] != filter_class:
                output_file.unlink()  # Delete the file
                filtered += 1
                continue
            
            processed += 1
            
            # Print result
            print(f"✓ {img_file.name}")
            print(f"  Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
            
        except Exception as e:
            print(f"✗ Error processing {img_file.name}: {e}")
    
    print("=" * 60)
    print(f"\nProcessed {processed} images")
    if filter_class:
        print(f"Filtered out {filtered} images (not '{filter_class}')")
    print(f"Visualizations saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate batch GradCAM visualizations for charter images'
    )
    parser.add_argument('--directory', type=str, required=True,
                       help='Directory containing images to visualize')
    parser.add_argument('--output', type=str, required=True,
                       help='Directory to save visualizations')
    parser.add_argument('--model', type=str, default='results/best_model.pth',
                       help='Path to trained model (default: results/best_model.pth)')
    parser.add_argument('--filter-class', type=str,
                       help='Only visualize images predicted as this class')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    batch_visualize(
        args.directory,
        args.output,
        model_path=args.model,
        filter_class=args.filter_class,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
