import json

def display_comparison():
    with open("results_fe.json", "r") as f:
        fe = json.load(f)
    with open("results_ft.json", "r") as f:
        ft = json.load(f)

    print("=" * 72)
    print(f"{'Stage':<22} | {'Val Accuracy':<14} | {'Train Time':<12} | {'Trainable Params':<16}")
    print("-" * 72)
    print(f"{'Feature Extraction':<22} | {fe['val_acc']:<14.4f} | {fe['train_time']:<10.2f}s | {fe['trainable_params']:<16}")
    print(f"{'Fine-Tuning':<22} | {ft['val_acc']:<14.4f} | {ft['train_time']:<10.2f}s | {ft['trainable_params']:<16}")
    print("=" * 72)

    diff = ft['val_acc'] - fe['val_acc']
    if diff > 0:
        print(f"\nFine-tuning outperformed feature extraction by {diff * 100:.2f}%.")
        print("Mechanism: Adapting layer4's high-level spatial receptive fields allowed")
        print("the model to learn domain-specific discriminative features (avian feathers")
        print("vs aircraft fuselages) rather than relying solely on generic ImageNet features.")
    elif diff < 0:
        print(f"\nFeature extraction outperformed fine-tuning by {abs(diff) * 100:.2f}%.")
        print("Mechanism: On a small training dataset, updating layer4 introduced")
        print("overfitting, whereas frozen ImageNet features generalized better.")
    else:
        print("\nBoth methods achieved identical accuracy.")
        print("Mechanism: The frozen ImageNet features were already sufficiently separable")
        print("for this target subset, so extra parameter updates yielded no marginal gain.")

if __name__ == "__main__":
    display_comparison()