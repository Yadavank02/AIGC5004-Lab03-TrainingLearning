import time
import json
import torch
import torch.nn as nn
from torchvision import models
from dataset import get_cifar_subsets

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def run_feature_extraction():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_cifar_subsets()

    # Pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

    # 1. Freeze every parameter in the backbone
    for param in model.parameters():
        param.requires_grad = False

    # 2. Replace head for 2 classes (fc parameters require gradients by default)
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

    trainable_params = count_trainable(model)
    print(f"Feature Extraction - Trainable parameters: {trainable_params}")

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    start_time = time.time()

    for epoch in range(epochs):
        # BatchNorm Rule: Backbone MUST remain in eval() mode
        model.eval()
        model.fc.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time
    val_acc = evaluate(model, val_loader, device)

    print(f"Feature Extraction Complete | Acc: {val_acc:.4f} | Time: {train_time:.2f}s")

    results = {
        "val_acc": val_acc,
        "train_time": train_time,
        "trainable_params": trainable_params
    }
    with open("results_fe.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_feature_extraction()