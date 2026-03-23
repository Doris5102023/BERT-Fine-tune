# 📝 BERT Fine-Tuning for SST-2
This project implements the fine-tuning of BERT and DistilBERT on the **SST-2 (Stanford Sentiment Treebank)** binary sentiment classification task, with complete data processing, training, evaluation, and result analysis.

---

## 🧠 What is BERT?
BERT (**Bidirectional Encoder Representations from Transformers**) is a pre-trained language model based on the Transformer encoder architecture. It learns deep contextualized word representations by jointly conditioning on both left and right context in all layers.

### 🔍 Core Features of BERT
| Feature | Description |
|---------|-------------|
| 🔄 Bidirectional | Captures context from both directions, enabling deeper language understanding compared to unidirectional models. |
| 🧠 Transformer Encoder | Uses multi-head self-attention to model dependencies between words regardless of their position. |
| 📚 Pre-trained & Fine-tuned | First pre-trained on large-scale corpora with general language objectives; then fine-tuned on downstream tasks. |
| 🥞 Layered Architecture | BERT-base consists of 12 transformer layers, enabling hierarchical feature extraction. |

### 📌 Pre-training Tasks of BERT
1. **Masked Language Modeling (MLM)**
   Randomly masks a portion of input tokens and trains the model to predict the masked tokens, enabling strong contextual understanding.

2. **Next Sentence Prediction (NSP)**
   Trains the model to classify whether two sentences are consecutive in the original corpus, helping capture inter-sentence relationships.

---

## 🔧 What is Fine-Tuning?
Fine-tuning adapts a **pre-trained BERT model** to a specific downstream task (e.g., sentiment classification) using task‑labeled data. The model retains general linguistic knowledge and adjusts its parameters slightly to fit task distribution.

### 🎯 Why Fine-Tuning?
- Leverages powerful pre-trained contextual representations.
- Requires significantly less data and computation than training from scratch.
- Achieves strong performance on most NLP tasks with minimal modification.

### 🧩 Fine-Tuning Pipeline for Sentiment Analysis
```
flowchart LR
    A[Pre-trained BERT] --> B[Add Classification Head]
    B --> C[Feed SST-2 Data]
    C --> D[Update Model Parameters]
    D --> E[Fine-tuned Sentiment Classifier]
```

### 📎 Key Fine-Tuning Settings
- Small learning rate (typically 2e−5) to avoid catastrophic forgetting.
- Moderate batch size for stable optimization.
- Limited epochs (3–5) to prevent overfitting.
- Lightweight variants (e.g., DistilBERT) for efficiency.

---

## 🚀 How to Use This Project
### 📋 Dependencies
```bash
pip install torch transformers datasets pandas numpy
```

### 📂 Dataset Preparation
1. Place the SST-2 dataset (`train.tsv`, `dev.tsv`, `test.tsv`) into `./data/SST2`.
2. Set `DATA_DIR` in the script to match your local path.

### 🏃 Run Training & Evaluation
```bash
python main.py
```

### 📊 Outputs
- Test-set accuracy and F1 scores.
- Results exported to `SST2_finetune_results.csv`.
- Model comparison between BERT and DistilBERT.
- Trained model checkpoints saved to `./output/`.

---

## 📊 Comparison: BERT-base vs DistilBERT
| Metric | BERT-base | DistilBERT |
|--------|-----------|-------------|
| ⏱️ Training Time | Longer | Shorter (≈40–50% faster) |
| 🧠 Parameters | 110M | 66M |
| 🎯 Accuracy | Higher | Slightly lower but close |
| 💾 Memory Usage | Higher | Lower |

DistilBERT retains approximately 97% of BERT’s performance while being smaller, faster, and more efficient.

---

## 📌 Summary
This project provides a complete, reproducible pipeline for fine-tuning BERT‑family models on SST‑2 sentiment analysis. The experimental results show that:
- BERT‑base delivers higher accuracy.
- DistilBERT offers an excellent trade‑off between performance and efficiency.

Both models are widely used in real-world NLP systems for text classification applications.

---

## 📄 License
This project is for educational and academic use only.
