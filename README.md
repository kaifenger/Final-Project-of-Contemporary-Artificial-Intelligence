# Multimodal Sentiment Classification

## Project Overview
This project implements a multimodal sentiment classification model that predicts sentiment labels (positive, neutral, negative) from paired text and image data.

**Course:** Contemporary Artificial Intelligence  
**Assignment:** Project 5 - Multimodal Emotion Classification  
**GitHub Repository:** [To be added]

---

## Dataset
- **Training Set:** 4001 samples with labels
- **Test Set:** 512 samples without labels
- **Data Format:** Each sample has a unique GUID with corresponding text file (.txt) and image file (.jpg)
- **Labels:** positive, neutral, negative (3-class classification)

---

## Project Structure
```
project5/
├── data/                          # Original dataset
├── src/                           # Source code
│   ├── data_loader.py            # Data loading utilities
│   ├── dataset.py                # Custom Dataset class
│   ├── models/                   # Model implementations
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Prediction script
│   └── utils.py                  # Utility functions
├── configs/                       # Configuration files
├── experiments/                   # Experiment logs
├── results/                       # Results and visualizations
├── checkpoints/                   # Model checkpoints
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

---

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Git

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd project5

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Data Preparation
```bash
# Data statistics and split
python src/data_loader.py
```

### Training
```bash
# Train text-only baseline
python src/train.py --config configs/text_only.yaml

# Train image-only baseline
python src/train.py --config configs/image_only.yaml

# Train multimodal model
python src/train.py --config configs/multimodal.yaml
```

### Evaluation
```bash
python src/evaluate.py --model_path checkpoints/best_model.pth
```

### Prediction
```bash
python src/predict.py --model_path checkpoints/best_model.pth --output results/predictions/
```

---

## Model Architecture
[To be updated with specific architecture details]

---

## Experimental Results
[To be updated with validation results]

---

## References

### Code Repositories
1. [HuggingFace Transformers](https://github.com/huggingface/transformers) - Text encoder (BERT/RoBERTa)
2. [OpenAI CLIP](https://github.com/openai/CLIP) - Multimodal pre-training
3. [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) - Image encoder (ResNet/ViT)
4. [Multimodal Deep Learning](https://github.com/declare-lab/multimodal-deep-learning) - Fusion strategies
5. [EDA NLP](https://github.com/jasonwei20/eda_nlp) - Text data augmentation

### Papers
1. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021
2. Lu et al. "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations", NeurIPS 2019
3. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
4. Vaswani et al. "Attention Is All You Need", NeurIPS 2017
5. Wei et al. "EDA: Easy Data Augmentation Techniques", EMNLP 2019

---

## Version History
- **v0.1:** Project initialization and data processing
- [More versions to be added]

---

## Author
[Your Name]  
[Your Student ID]  
[Date: 2026-01-15]
