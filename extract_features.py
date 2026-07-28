import os
import json
import joblib
import numpy as np
import cv2
from pathlib import Path
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

PROCESSED_DIR = Path("Processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def extract_color_hsv_hist(img_bgr, h_bins=8, s_bins=8, v_bins=8):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_color_lab_hist(img_bgr, bins=16):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([lab], [0], None, [bins], [0, 256])
    hist_a = cv2.calcHist([lab], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([lab], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_l, hist_a, hist_b]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    return hist

def extract_lbp_texture(img_bgr, P=24, R=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    return hist

def extract_hog_features(img_bgr, target_size=(128, 128)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, target_size)
    hog_features = hog(
        gray_resized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return hog_features

def extract_image_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    hsv_feat = extract_color_hsv_hist(img)
    lab_feat = extract_color_lab_hist(img)
    lbp_feat = extract_lbp_texture(img)
    hog_feat = extract_hog_features(img)
    
    # Combined feature vector
    return np.concatenate([hsv_feat, lab_feat, lbp_feat, hog_feat])

def load_dataset_features(split_name, categories):
    X, y = [], []
    print(f"Extracting features for '{split_name}' split...")
    for class_idx, cat in enumerate(categories):
        cat_dir = PROCESSED_DIR / split_name / cat
        img_paths = list(cat_dir.glob("*.jpg"))
        for img_path in tqdm(img_paths, desc=f"{split_name}/{cat}"):
            try:
                feat = extract_image_features(img_path)
                X.append(feat)
                y.append(class_idx)
            except Exception as e:
                print(f"Error extracting {img_path}: {e}")
    return np.array(X), np.array(y)

def run_feature_extraction_pipeline():
    if not (PROCESSED_DIR / "dataset_summary.json").exists():
        raise FileNotFoundError("Processed dataset not found. Run preprocess_dataset.py first.")
        
    with open(PROCESSED_DIR / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    categories = summary["categories"]
    print(f"Categories ({len(categories)}): {categories}")
    
    X_train, y_train = load_dataset_features("train", categories)
    X_val, y_val = load_dataset_features("val", categories)
    X_test, y_test = load_dataset_features("test", categories)
    
    print(f"Train samples: {X_train.shape[0]} | Feature dimension: {X_train.shape[1]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="mlogloss", random_state=42)
    }
    
    best_model_name = None
    best_val_acc = -1.0
    best_model_obj = None
    
    results = {}
    
    for name, model in models.items():
        print(f"\n==========================================")
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        val_preds = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"{name} Validation Accuracy: {val_acc * 100:.2f}%")
        
        test_preds = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        print(f"{name} Test Accuracy:       {test_acc * 100:.2f}%")
        
        results[name] = {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "classification_report": classification_report(y_test, test_preds, target_names=categories, output_dict=True)
        }
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = name
            best_model_obj = model
            
    print(f"\n==========================================")
    print(f"Best Traditional CV Model: {best_model_name} (Val Acc: {best_val_acc * 100:.2f}%)")
    
    # Save model, scaler, and categories metadata
    artifact = {
        "model": best_model_obj,
        "scaler": scaler,
        "categories": categories,
        "model_name": best_model_name
    }
    
    joblib.dump(artifact, MODELS_DIR / "best_ml_model.pkl")
    print(f"Saved best ML model artifact to: {MODELS_DIR / 'best_ml_model.pkl'}")
    
    with open(MODELS_DIR / "ml_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_feature_extraction_pipeline()
