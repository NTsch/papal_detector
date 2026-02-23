"""
organizes charter images into train/validation/test splits
with proper directory structure for PyTorch training

example expected input structure:
images/
  papal/
  non_papal/

output structure:
data/
  train/
    papal/         (70% of images)
    non_papal/
  val/
    papal/         (15% of images)
    non_papal/
  test/
    papal/         (15% of images)
    non_papal/
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict


def prepare_data(source_dir='images', 
                 output_dir='data-split',
                 train_ratio=0.70,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 seed=42):
    """
    args:
        source_dir: dir containing class subdirectories
        output_dir: output directory for data
        train_ratio: proportion training set
        val_ratio: proportion validation set
        test_ratio: proportion test set
        seed: random seed
    """
    
    # ratios must sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # check source directory exists
    if not source_path.exists():
        raise ValueError(f"Source directory {source_dir} does not exist")
    
    print("=" * 60)
    print("Papal Charter Dataset Preparation")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    print(f"Random seed: {seed}")
    print("=" * 60)
    
    # create output dirs
    splits = ['train', 'val', 'test']
    classes = ['papal', 'non_papal', 'papal_canapis', 'non_papal_solemn']
    
    for split in splits:
        for class_name in classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # process each class
    stats = defaultdict(lambda: defaultdict(int))
    
    for class_name in classes:
        print(f"\nProcessing {class_name} charters...")
        
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} does not exist, skipping...")
            continue
        
        # get all image files
        image_files = []
        image_files.extend(list(class_dir.glob('*.jpg')))
        
        if not image_files:
            print(f"  Warning: No images found in {class_dir}")
            continue
        
        # shuffle imgs
        random.shuffle(image_files)
        
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # split imgs
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        print(f"  Total images: {total_images}")
        print(f"  Train: {len(train_images)}")
        print(f"  Val: {len(val_images)}")
        print(f"  Test: {len(test_images)}")
        
        # copy files to dirs
        for img_file in train_images:
            dest = output_path / 'train' / class_name / img_file.name
            shutil.copy2(img_file, dest)
            stats['train'][class_name] += 1
        
        for img_file in val_images:
            dest = output_path / 'val' / class_name / img_file.name
            shutil.copy2(img_file, dest)
            stats['val'][class_name] += 1
        
        for img_file in test_images:
            dest = output_path / 'test' / class_name / img_file.name
            shutil.copy2(img_file, dest)
            stats['test'][class_name] += 1
    
    # print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print("\nFinal dataset distribution:")
    for split in splits:
        print(f"\n{split.upper()}:")
        for class_name in classes:
            count = stats[split][class_name]
            print(f"  {class_name}: {count} images")
        total = sum(stats[split].values())
        print(f"  Total: {total} images")
    
    print("\n" + "=" * 60)
    print(f"Data prepared successfully in: {output_dir}/")
    print("You can now run: python train_classifier.py")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare papal charter dataset')
    parser.add_argument('--source', type=str, default='images',
                       help='Source directory containing class folders')
    parser.add_argument('--output', type=str, default='data_split',
                       help='Output directory for organized data')
    parser.add_argument('--train', type=float, default=0.70,
                       help='Training set ratio (default: 0.70)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    prepare_data(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
