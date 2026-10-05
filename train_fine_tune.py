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

def run_fine_tuning():
    torch.manual_seed(42)  # Identical seed to feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_cifar_subsets()

    # Pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze layer4 only
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 3. Replace head for 2 classes
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

    trainable_params = count_trainable(model)
    print(f"Fine-Tuning - Trainable parameters: {trainable_params}")

    # Discriminative learning rates via parameter groups
    optimizer = torch.optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 1e-3},       # Head trains faster
        {'params': model.layer4.parameters(), 'lr': 1e-5},   # Backbone adapts conservatively
    ])
    criterion = nn.CrossEntropyLoss()

    epochs = 10  # Identical epoch count
    start_time = time.time()

    for epoch in range(epochs):
        model.train()  # Normal train mode is active since layer4 is adapting

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time
    val_acc = evaluate(model, val_loader, device)

    print(f"Fine-Tuning Complete | Acc: {val_acc:.4f} | Time: {train_time:.2f}s")

    results = {
        "val_acc": val_acc,
        "train_time": train_time,
        "trainable_params": trainable_params
    }
    with open("results_ft.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_fine_tuning()