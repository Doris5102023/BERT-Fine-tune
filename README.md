# 🐻 BERT Fine-Tuning for SST-2 😊
An easy guide to BERT + fine-tuning, with code to fine-tune BERT/DistilBERT on SST-2 dataset!

---

## 🎀 What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is like a **multilingual bear** who read tons of books (text data) and learned to understand words in context!

### 🧸 Core Features of BERT
| Feature | Explanation |
|---------|------------------|
| 🔄 Bidirectional | Unlike a one-way train 🚂, BERT reads words from LEFT → RIGHT AND RIGHT → LEFT (understands "apple" in "eat apple" vs "apple phone"!) |
| 🧠 Transformer Encoder | BERT’s brain 🧠—uses "self-attention" to hug 🫂 important words (e.g., in "I love this movie", it focuses on "love"!) |
| 📚 Pre-trained + Fine-tuned | First learns from big books (pre-training), then studies small workbooks (fine-tuning) for specific tasks (like sentiment analysis!) |
| 🥞 Layered Structure | BERT-base has 12 layers (like 12 pancakes 🥞)—each layer learns deeper meaning! |

### 🍪 How BERT Learns (Pre-training Tasks)
1. **Masked Language Modeling (MLM)**  
   Hide some words (e.g., "I [MASK] this movie") and let BERT guess the hidden word → like fill-in-the-blank games 🎮!
2. **Next Sentence Prediction (NSP)**  
   Let BERT guess if two sentences are related (e.g., "I ate breakfast" → "I drank milk" = YES ✅; "I ate breakfast" → "Mars is red" = NO ❌)

---

## 🐾 What is Fine-Tuning? (Like Teaching a Bear New Tricks!)
Fine-tuning is when we take the pre-trained BERT (smart bear 🐻) and teach it to do a **specific task** (e.g., sentiment analysis) with small task-specific data!

### 🎯 Why Fine-Tuning?
- Pre-trained BERT knows general language rules (like a bear who knows all forest languages 🌲)
- Fine-tuning adapts BERT to your task (teach the bear to recognize "happy" vs "sad" sentences 😊😢)
- Saves time: no need to train a model from scratch (like teaching a bear to dance 💃 instead of teaching it to walk first!)

### 🍡 Fine-Tuning Process for Sentiment Analysis
```
flowchart LR
    A[Pre-trained BERT 🐻] --> B[Add Task Head 🎩] (Add a small neural network for classification)
    B --> C[Feed SST-2 Data 📄] (Sentences + labels: 0=negative 😞, 1=positive 😊)
    C --> D[Adjust Weights 🪜] (Tweak BERT’s brain a little—don’t forget old knowledge!)
    D --> E[Trained Model 🐼] (Now BERT can judge if a sentence is happy/sad!)
```

### 🧁 Key Tips for Fine-Tuning
1. **Low Learning Rate** → Like feeding the bear small honey drops 🍯 (don’t overwhelm it!)
2. **Small Batch Size** → Teach the bear 16 sentences at a time (not 1000!)
3. **Few Epochs** → Train for 3-5 rounds (the bear gets bored if trained too long 🥱)
4. **DistilBERT** → A smaller bear 🐹 (6 layers) with 90% of BERT’s ability—faster!

---

## 🚀 How to Use This Code?
### 📋 Requirements
```bash
pip install torch transformers datasets pandas numpy
```

### 📂 Dataset Setup
1. Put SST-2 dataset (train.tsv/dev.tsv/test.tsv) in `./data/SST2` folder 📁
2. Modify `DATA_DIR` in the code to your dataset path

### 🏃 Run the Code
```bash
python main.py
```

### 📊 Output
- A CSV file (`SST2_finetune_results.csv`) with accuracy/F1 scores 📈
- Console output with model comparison (BERT vs DistilBERT)
- Trained models saved in `./output/` folder 📦

---

## 🐻 Comparison: BERT vs DistilBERT
| Metric | BERT-base 🐻 | DistilBERT 🐹 |
|--------|--------------|---------------|
| ⏱️ Training Time | Slow (like a bear walking 🐢) | Fast (like a squirrel running 🐿️) |
| 🧠 Parameters | 110M (big brain) | 66M (small but smart brain) |
| 🎯 Accuracy | Higher (90%+) | Slightly lower (88%+) but close! |
| 💾 Memory | More (needs big backpack 🎒) | Less (fits in small pocket 🧺) |

---

## 🎨 Fun Facts
- BERT was created by Google in 2018 (like a teddy bear 🧸 from Google Store!)
- DistilBERT is a "distilled" version of BERT (like mini bear candies 🍬)
- Fine-tuning BERT on SST-2 takes ~5 mins on GPU (enough time to eat a popsicle 🍧!)

---

## ✨ License
This code is as free as a bear playing in the forest 🌳—use it for homework/learning!

---
