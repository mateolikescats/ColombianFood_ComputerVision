import os
import json
import random
from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif

# Register HEIC opener for PIL
pillow_heif.register_heif_opener()

# Configuration
RAW_DIR = Path("Raw_source")
PROCESSED_DIR = Path("Processed")
TARGET_SIZE = (512, 512)
TRAIN_RATIO = 0.69
VAL_RATIO = 0.20
TEST_RATIO = 0.11
RANDOM_SEED = 42

def load_and_standardize_image(file_path: Path, category_name: str) -> Image.Image:
    """
    Loads an image file (supporting JPG, PNG, HEIC, WEBP, etc.), fixes orientation,
    applies custom square cropping, and resizes to TARGET_SIZE (512x512).
    """
    img = Image.open(file_path)
    
    # Auto-rotate based on EXIF orientation metadata
    img = ImageOps.exif_transpose(img)
    
    # Convert to RGB color mode
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    width, height = img.size
    min_dim = min(width, height)
    
    is_heic = file_path.suffix.lower() in [".heic", ".heif"]
    is_almojabana = category_name.lower() == "almojabana"
    
    # Custom cropping logic
    if is_almojabana and is_heic:
        # HEIC Almojabana images have subject lower in the frame -> crop lower region
        x_start = (width - min_dim) // 2
        y_start = max(0, height - min_dim)
    else:
        # Default center crop
        x_start = (width - min_dim) // 2
        y_start = (height - min_dim) // 2
        
    crop_box = (x_start, y_start, x_start + min_dim, y_start + min_dim)
    cropped_img = img.crop(crop_box)
    
    # Resize with high-quality anti-aliasing (LANCZOS)
    resized_img = cropped_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    return resized_img


def process_dataset():
    random.seed(RANDOM_SEED)
    
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Source directory '{RAW_DIR}' does not exist.")
        
    categories = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    categories.sort()
    
    print(f"Found {len(categories)} categories: {categories}")
    
    summary = {
        "target_resolution": list(TARGET_SIZE),
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "categories": categories,
        "counts": {}
    }
    
    # Create Processed directory structure
    for split in ["train", "val", "test"]:
        for cat in categories:
            (PROCESSED_DIR / split / cat).mkdir(parents=True, exist_ok=True)
            
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
    
    total_processed = 0
    total_augmented = 0

    for cat in categories:
        cat_dir = RAW_DIR / cat
        all_files = [
            f for f in cat_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        # Shuffle for stratified split reproducibility
        random.shuffle(all_files)
        
        num_files = len(all_files)
        if num_files == 0:
            print(f"Warning: Category '{cat}' has no valid images.")
            continue
            
        n_train = max(1, int(num_files * TRAIN_RATIO))
        n_val = max(1, int(num_files * VAL_RATIO))
        # Ensure remaining files go to test
        n_test = num_files - n_train - n_val
        if n_test <= 0 and num_files >= 3:
            n_train -= 1
            n_test = 1
            
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        splits_dict = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        cat_stats = {"raw_total": num_files, "splits": {}}
        
        for split, file_list in splits_dict.items():
            saved_count = 0
            aug_count = 0
            
            for idx, file_path in enumerate(file_list):
                try:
                    standard_img = load_and_standardize_image(file_path, cat)
                    
                    base_name = f"{cat}_{idx+1:04d}.jpg"
                    out_path = PROCESSED_DIR / split / cat / base_name
                    standard_img.save(out_path, format="JPEG", quality=95)
                    saved_count += 1
                    total_processed += 1
                    
                    # Apply horizontal mirror flip ONLY to train split
                    if split == "train":
                        flipped_img = standard_img.transpose(Image.FLIP_LEFT_RIGHT)
                        aug_name = f"{cat}_{idx+1:04d}_flipped.jpg"
                        aug_out_path = PROCESSED_DIR / split / cat / aug_name
                        flipped_img.save(aug_out_path, format="JPEG", quality=95)
                        aug_count += 1
                        total_augmented += 1
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
            cat_stats["splits"][split] = {
                "original": saved_count,
                "augmented": aug_count,
                "total": saved_count + aug_count
            }
            
        summary["counts"][cat] = cat_stats
        print(f"Category '{cat}': {num_files} raw -> "
              f"Train: {cat_stats['splits']['train']['total']} (incl. {cat_stats['splits']['train']['augmented']} flips), "
              f"Val: {cat_stats['splits']['val']['original']}, "
              f"Test: {cat_stats['splits']['test']['original']}")

    summary["totals"] = {
        "original_processed": total_processed,
        "augmented": total_augmented,
        "total_images_in_processed": total_processed + total_augmented
    }
    
    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("\nDataset processing completed successfully!")
    print(f"Total processed original images: {total_processed}")
    print(f"Total augmented (flipped) images: {total_augmented}")
    print(f"Dataset summary saved to: {PROCESSED_DIR / 'dataset_summary.json'}")

if __name__ == "__main__":
    process_dataset()
