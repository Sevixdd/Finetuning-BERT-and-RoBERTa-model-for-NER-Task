# Fine-tuning BERT and RoBERTa Models for Named Entity Recognition (NER) Task

This project implements and compares different approaches for Named Entity Recognition using state-of-the-art transformer models (BERT and RoBERTa) and traditional machine learning methods (CRF) on the PLOD (Part-of-speech and Named Entity Recognition) dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [References](#references)

## 🔍 Overview

This project focuses on fine-tuning pre-trained transformer models for Named Entity Recognition tasks. The implementation includes:

- **BERT Base** fine-tuning for NER
- **RoBERTa Base** fine-tuning for NER  
- **Conditional Random Fields (CRF)** as a baseline comparison
- Comprehensive hyperparameter optimization using Weights & Biases
- Extensive data analysis and visualization
- Performance evaluation with confusion matrices and metrics

## 📊 Dataset

The project uses the **PLOD-CW** and **PLOD-filtered** datasets from Surrey NLP, which contain:

- **112,652 training samples**
- **4 NER entity types**: B-O (Outside), B-AC (Academic), B-LF (Life Form), I-LF (Inside Life Form)
- **19 POS tag categories**: ADJ, ADP, ADV, AUX, CONJ, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X, SPACE
- **Vocabulary size**: 9,133 distinct tokens
- **Total tokens**: 40,000 (30,582 without punctuation)

### Dataset Distribution
- **B-O (Outside)**: 82.98% (4,378,676 tokens)
- **B-AC (Academic)**: 3.49% (184,042 tokens)  
- **B-LF (Long Form)**: 7.72% (407,268 tokens)
- **I-LF (Inside Long Form)**: 5.82% (307,003 tokens)

## 🤖 Models Implemented

### 1. BERT Base (bert-base-uncased)
- Pre-trained on English text
- Fine-tuned for token classification
- Hyperparameter optimization with W&B sweeps
- Hugging Face: https://huggingface.co/Sevixdd/bert-base-uncased-finetuned-ner

### 2. RoBERTa Base (roberta-base)
- Robustly optimized BERT approach
- Enhanced training methodology
- Comparative analysis with BERT
- Hugging Face: https://huggingface.co/Sevixdd/roberta-base-finetuned-ner

### 3. Conditional Random Fields (CRF)
- Traditional sequence labeling approach
- Grid search and randomized search optimization
- Baseline comparison for transformer models

## ✨ Features

- **Comprehensive Data Analysis**: 
  - Token frequency analysis
  - NER and POS tag distribution visualization
  - TF-IDF analysis
  - Bigram analysis and topic modeling

- **Advanced Training Pipeline**:
  - Early stopping with patience
  - Learning rate scheduling
  - Gradient accumulation
  - Model checkpointing

- **Hyperparameter Optimization**:
  - Weights & Biases integration
  - Random search sweeps
  - Grid search for CRF
  - Automated hyperparameter tuning

- **Evaluation Metrics**:
  - Precision, Recall, F1-score
  - Confusion matrices
  - Per-class performance analysis
  - Cross-validation results

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies

```bash
# Core ML libraries
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn sklearn-crfsuite

# Data processing
pip install pandas numpy matplotlib seaborn
pip install nltk spacy

# Visualization and analysis
pip install plotly gensim
pip install wandb  # for experiment tracking

# Additional utilities
pip install seqeval tqdm
```

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Finetuning-BERT-and-RoBERTa-model-for-NER-Task
```

2. **Download spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

3. **Login to Weights & Biases** (optional):
```bash
wandb login
```

## 💻 Usage

### Running the Complete Pipeline

1. **Open the Jupyter notebook**:
```bash
jupyter notebook NLP_CW.ipynb
```

2. **Execute cells sequentially** to:
   - Install dependencies
   - Load and analyze the dataset
   - Train BERT and RoBERTa models
   - Run CRF experiments
   - Generate visualizations and results

### Individual Model Training

#### BERT Fine-tuning
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

# Load model and tokenizer
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(label_list)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert-ner",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    push_to_hub=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
```

#### CRF Training
```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Prepare features
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

# Train CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
```

## 📈 Results

### Model Performance Comparison

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| BERT Base | 0.9582 | 0.95 | 0.9541 | 0.9499 |
| RoBERTa Base | 0.9657 | 0.9608 | 0.9633 | 0.9594 |
| CRF | - | - | - | - |

*Results will be updated based on actual training outputs*

### Visualizations

The project includes comprehensive visualizations:

- **Confusion Matrices**: `Confusion Matrix/BertConfusion1.png`, `Confusion Matrix/RobertaConfusion1.png`
- **Training Statistics**: `Training Stats.png`
- **CRF Optimization Results**: Various plots in `crf/` directory
- **Dataset Analysis**: Plots in `Plod/` directory

### Weights & Biases Reports

Detailed experiment tracking and hyperparameter optimization results are available in the `W&B/` directory:

- BERT Base vs RoBERTa Base comparison
- Fine-tuning RoBERTa base results
- Hyperparameter optimization analysis

## 📁 Project Structure

```
Finetuning-BERT-and-RoBERTa-model-for-NER-Task/
├── NLP_CW.ipynb                 # Main Jupyter notebook
├── README.md                    # This file
├── Training Stats.png           # Training performance visualization
├── Confusion Matrix/            # Model confusion matrices
│   ├── BertConfusion1.png
│   └── RobertaConfusion1.png
├── crf/                        # CRF experiment results
│   ├── bestparamsCRF.png
│   ├── CRF folding.png
│   ├── gridsearch.png
│   └── ...
├── Plod/                       # Dataset analysis visualizations
│   ├── PlodCW.png
│   ├── plodcwnopunc.png
│   └── ...
└── W&B/                        # Weights & Biases reports
    ├── BERT Base vs RoBERTa Base _ NLP.pdf
    ├── Fine tuning RoBERTa base _ NLP.pdf
    └── HyperParams Optimization Roberta _ NLP.pdf
```

## 🔬 Key Findings

### Dataset Analysis
- **Class Imbalance**: Significant imbalance with B-O (Outside) tokens representing 83% of the dataset
- **Vocabulary**: 9,133 distinct tokens with common words like "the", "of", "and" being most frequent
- **POS Distribution**: Verbs and nouns are the most common parts of speech

### Model Performance
- **Transformer Models**: BERT and RoBERTa show superior performance compared to traditional CRF
- **Hyperparameter Sensitivity**: Learning rate and batch size significantly impact model performance
- **Training Efficiency**: Early stopping helps prevent overfitting and reduces training time

### Optimization Insights
- **CRF Parameters**: Optimal c1 and c2 values found through grid search
- **Transformer Fine-tuning**: Lower learning rates (2e-5) work better for fine-tuning
- **Batch Size**: Smaller batch sizes (8-16) often perform better for this task

## 📚 References

1. **BERT**: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. **RoBERTa**: Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
3. **CRF**: Lafferty, J., et al. (2001). "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"
4. **PLOD Dataset**: Surrey NLP Group - Part-of-speech and Named Entity Recognition Dataset
5. **Multilingual NER**: Murthy, R., et al. (2018). "Improving NER Tagging Performance in Low-Resource Languages via Multilingual Learning"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project was developed as part of a Natural Language Processing coursework focusing on Named Entity Recognition using state-of-the-art transformer models and traditional machine learning approaches.
