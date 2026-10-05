import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 1. Training transform with label-preserving augmentations
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 2. Evaluation transform for a fixed, deterministic view
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Custom dataset wrapper to apply separate transforms per split
class SubsetImageDataset(Dataset):
    def __init__(self, base_dataset, indices, transform, label_map):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[label]

# 4. Function to create balanced CIFAR-10 data loaders
def get_cifar_subsets(samples_per_class_train=100, samples_per_class_val=50):
    raw_train = datasets.CIFAR10(root="./data", train=True, download=True)
    raw_val = datasets.CIFAR10(root="./data", train=False, download=True)

    target_classes = [0, 2]  # Class 0: Airplane, Class 2: Bird
    label_map = {0: 0, 2: 1}  # Map to 0 and 1 for CrossEntropyLoss

    def filter_samples(dataset, target_classes, limit):
        indices = []
        counts = {c: 0 for c in target_classes}
        for i, (_, label) in enumerate(dataset):
            if label in target_classes and counts[label] < limit:
                indices.append(i)
                counts[label] += 1
            if all(counts[c] >= limit for c in target_classes):
                break
        return indices

    train_idx = filter_samples(raw_train, target_classes, samples_per_class_train)
    val_idx = filter_samples(raw_val, target_classes, samples_per_class_val)

    train_data = SubsetImageDataset(raw_train, train_idx, train_transform, label_map)
    val_data = SubsetImageDataset(raw_val, val_idx, eval_transform, label_map)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    return train_loader, val_loader