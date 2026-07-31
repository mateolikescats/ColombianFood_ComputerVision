import os
import json
import argparse
from pathlib import Path
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("Processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Silence Optuna verbose logs for cleaner console output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_dataloaders(batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3), # Light augmentation during fine-tuning
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(PROCESSED_DIR / x, data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'val', 'test']
    }
    
    return dataloaders, image_datasets

def build_model(model_name, num_classes, dropout_rate=0.2):
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout_rate)
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optuna Search Space
    model_name = trial.suggest_categorical("model_name", ["mobilenet_v3_small", "resnet18"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])

    dataloaders, image_datasets = get_dataloaders(batch_size)
    num_classes = len(image_datasets['train'].classes)

    model = build_model(model_name, num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    epochs_per_trial = 10
    best_trial_val_acc = 0.0

    for epoch in range(epochs_per_trial):
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = (val_corrects.double() / len(image_datasets['val'])).item()
        if val_acc > best_trial_val_acc:
            best_trial_val_acc = val_acc

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_trial_val_acc

def train_final_model(best_params, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Training Final Optimized CNN for {epochs} Epochs")
    print(f"Device: {device}")
    print(f"Hyperparameters: {json.dumps(best_params, indent=2)}")
    print(f"==========================================")

    dataloaders, image_datasets = get_dataloaders(best_params['batch_size'])
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    model = build_model(best_params['model_name'], num_classes, best_params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()

    if best_params['optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], momentum=0.9, weight_decay=best_params['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "best_cnn_model.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = (running_corrects.double() / len(image_datasets[phase])).item()

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.2f}%")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_name': best_params['model_name'],
                    'state_dict': model.state_dict(),
                    'class_names': class_names,
                    'val_acc': best_val_acc,
                    'best_params': best_params
                }, best_model_path)
                print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc*100:.2f}%) to {best_model_path}")

    # Evaluate Final Model on Test Set
    print("\n==========================================")
    print("Evaluating Optimized CNN on Test Set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Set Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    optuna_results = {
        "best_params": best_params,
        "epochs": epochs,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "history": history
    }

    with open(MODELS_DIR / "cnn_optuna_results.json", "w", encoding="utf-8") as f:
        json.dump(optuna_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {MODELS_DIR / 'cnn_optuna_results.json'}")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization & Retraining for Colombian Food CNN")
    parser.add_argument("--n_trials", type=int, default=8, help="Number of Optuna trials")
    parser.add_argument("--final_epochs", type=int, default=25, help="Number of epochs for final training")
    args = parser.parse_args()

    print(f"Starting Optuna Hyperparameter Study ({args.n_trials} trials)...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials)

    print("\n==========================================")
    print("Optuna Hyperparameter Optimization Complete!")
    print(f"Best Trial Val Accuracy: {study.best_value * 100:.2f}%")
    print(f"Best Hyperparameters: {json.dumps(study.best_params, indent=2)}")

    train_final_model(study.best_params, epochs=args.final_epochs)


if __name__ == "__main__":
    main()
