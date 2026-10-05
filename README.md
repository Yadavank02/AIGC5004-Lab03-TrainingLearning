# From Frozen Features to Fine-Tuning
Fine-tuning a pretrained ResNet-18 vision backbone on a small CIFAR-10 dataset (Airplanes vs. Birds) to compare feature extraction against fine-tuning[cite: 13, 15].

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run
Bash
python train_feature_extraction.py
python train_fine_tune.py
python compare.py

Dataset
We picked two categories from CIFAR-10—airplanes and birds—with an even split between them. That gives us 200 images for training (100 per class) and 100 for validation (50 per class). Because the original images are a tiny 32x32, we scaled them up to 256 and center-cropped to 224x224 to match what ResNet-18 needs.

Results
Feature extraction gave us 91% validation accuracy in 28.3 seconds while only training 1,026 parameters (the rest of the backbone stayed frozen). Fine-tuning layer 4 bumped our trainable parameters up to nearly 8.4 million and took about 35 seconds, but the accuracy ended up lower at 84%.

Conclusion
Feature extraction won by 7 percentage points (91% vs. 84%), and it mostly comes down to dataset size. We only had 200 training examples, so opening up 8.4 million weights in layer 4 made the model overfit almost immediately. ResNet-18 already had great features for objects like birds and airplanes from its ImageNet pretraining, so freezing the backbone and updating only the 1,026 weights in the final layer worked much better. With tiny datasets, keeping it simple with feature extraction is usually the way to go.

Known Limitations
For starters, stretching tiny 32x32 images up to 224x224 makes them fuzzy without adding any useful info. On top of that, trying to fine-tune deeper layers with only 200 training pictures just doesn't give the model enough data to learn properly without memorizing the inputs.

