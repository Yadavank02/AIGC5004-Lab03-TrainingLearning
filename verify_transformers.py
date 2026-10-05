import torch
from torchvision import datasets
from dataset import train_transform, eval_transform

def test_transforms():
    raw_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    sample_img, _ = raw_dataset[0]

    # Test eval_transform 5 times: must be completely identical
    eval_passes = [eval_transform(sample_img) for _ in range(5)]
    all_eval_identical = True
    for i in range(1, 5):
        if not torch.allclose(eval_passes[0], eval_passes[i]):
            all_eval_identical = False
            break

    # Test train_transform 5 times: must have random differences
    train_passes = [train_transform(sample_img) for _ in range(5)]
    any_train_difference = False
    for i in range(1, 5):
        if not torch.allclose(train_passes[0], train_passes[i]):
            any_train_difference = True
            break

    print(f"Eval transform identical across 5 passes: {all_eval_identical}")
    print(f"Train transform creates variations across passes: {any_train_difference}")

    assert all_eval_identical, "Error: eval_transform is not deterministic!"
    assert any_train_difference, "Error: train_transform produced zero augmentation!"
    print("Verification passed: Part 3 transform rules validated.")

if __name__ == "__main__":
    test_transforms()