# Anime Image Classifier

A robust deep learning project that classifies over 130+ anime characters with high accuracy. This project demonstrates a complete end-to-end ML pipeline: creating a custom dataset, implementing transfer learning with ResNet34, optimizing hyperparameters with Optuna, and fine-tuning for production-ready performance.

## Model Architecture

This project utilizes **Transfer Learning** to leverage the power of pre-trained networks.

### The Backbone: ResNet34

We use **ResNet34** (Residual Network with 34 layers) pre-trained on ImageNet as our backbone.

- **Why ResNet34?** It offers an excellent balance between performance and computational efficiency. Unlike deeper models (ResNet50/101), it is faster to train and less prone to overfitting on medium-sized custom datasets.
- **Feature Extraction:** The backbone extracts high-level features (shapes, textures, patterns) from the anime images.

### The Custom Head

The original ImageNet classification layer is replaced with a custom `ClassificationHead` designed for our specific task:

1.  **Linear Layer**: Projects features to a hidden dimension (e.g., 512).
2.  **LayerNorm**: Stabilizes training by normalizing the inputs across the features.
3.  **GELU Activation**: A modern activation function (Gaussian Error Linear Unit) that often outperforms ReLU in deeper networks.
4.  **Dropout**: Prevents overfitting by randomly zeroing out neurons during training.
5.  **L2 Normalization**: The output embedding is normalized before the final projection.
6.  **Final Linear Layer**: Maps to the number of classes (130+ characters).

### Two-Stage Training Strategy

1.  **Linear Probing (Stage 1)**: We freeze the ResNet backbone and only train the custom head. This allows the head to learn to use the pre-trained features without destroying them.
2.  **Fine-Tuning (Stage 2)**: We unfreeze the deeper layers (specifically `layer4`) of the backbone. This allows the model to adapt its feature extraction specifically for anime art styles, which differ significantly from real-world ImageNet photos.

---

## Project Flow

1.  **Data Setup**: Utilizes the [Kaggle Anime Face Dataset](https://www.kaggle.com/datasets/thedevastator/anime-face-dataset-by-character-name?resource=download). `split_data.py` handles cleaning, deduplication (via `imagehash`), and stratified splitting.
2.  **Augmentation & Transforms**: `src/data/transforms.py` defines image preprocessing (resizing, normalization) and augmentations (random crops, flips) to improve model robustness.
3.  **Data Loading**: `src/data/dataloader.py` creates efficient, GPU-safe PyTorch DataLoaders that apply the defined transforms on the fly.
4.  **Optimization**: `src/optuna.py` finds optimal hyperparameters (LR, batch size, etc.).
5.  **Training**:
    - **Linear Probing**: `src/train.py` trains the custom classifier while freezing the pre-trained backbone.
    - **Fine-Tuning**: `src/fine_tuning.py` unfreezes `layer4` (which contains high-level semantic features) to adapt the backbone to anime art styles.
6.  **Evaluation**: `src/evaluate.py` generates detailed per-class performance reports.

---

## Getting Started

Follow these steps to clone the repository and run the project locally.

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (Recommended for training)

### 1. Clone the Repository

```bash
git clone https://github.com/KhanUzeb/Anime-Image-Classifier.git
cd Anime-Image-Classifier
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Data Setup

First, prepare the project directories:

```bash
mkdir -p data/raw data/processed checkpoints
```

Next, place your raw image folders in `data/raw/` (e.g., `data/raw/naruto/`, `data/raw/sasuke/`).

Then run the processing script:

```bash
python split_data.py
```

This will generate the `data/processed/` folder.

### 4. Training

**Option A: Tune & Train (Recommended)**
Run Optuna to find the best hyperparameters and automatically train the model.

```bash
python src/optuna.py
```

**Option B: Manual Training**
Train the head manually:

```bash
python src/train.py
```

Then fine-tune the backbone:

```bash
python src/fine_tuning.py
```

### 5. Evaluation

Generate a performance report on the test set:

```bash
python src/evaluate.py
```

The report will be saved to `reports/`.

---

## Results

| Stage                | Accuracy | Weighted F1 |
| -------------------- | -------- | ----------- |
| Baseline (Head Only) | ~52%     | 0.50        |
| **Fine-Tuned**       | **~77%** | **0.77**    |

The model demonstrates strong performance on key characters with 90%+ precision on distinctive designs.
