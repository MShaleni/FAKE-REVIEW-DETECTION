# Fake Review Detection Using Emotion-Aware RoBERTa with SHAP Explainability

This project builds an emotion-aware system for detecting fake product reviews by combining a fine-tuned RoBERTa transformer with lexicon-based emotion features and SHAP interpretability.  
The goal is to improve the reliability and transparency of fake review classification by analyzing both **contextual meaning** and **emotional patterns** in text.  
The work is part of an academic research project completed at VIT Chennai.  
 

---

##  1. Project Overview

Online reviews heavily influence buying decisions, but a significant portion of them are deceptive. This system detects fake reviews by using:

- **RoBERTa transformer model** for deep contextual understanding  
- **NRC Emotion Lexicon** to extract eight emotions  
- **Late-fusion feature architecture** combining semantic + emotional signals  
- **SHAP explainability** to highlight important tokens influencing predictions  

The model shows strong performance and provides transparent explanations suitable for academic, research, and real-world e-commerce use cases.  
 
---

##  2. Key Features

### -> **Emotion-Aware Transformer Model**
Integrates eight normalized emotion scores (anger, fear, anticipation, trust, surprise, sadness, joy, disgust) with RoBERTa embeddings.

### -> **Late Fusion Strategy**
Combines RoBERTa output with emotion vectors to improve sensitivity to deceptive patterns.

### -> **Explainability with SHAP**
Generates token-level explanations showing why the model predicted “fake” or “real”.

### -> **Strong Evaluation Performance**
- **Accuracy:** 0.844  
- **Precision:** 0.936  
- **Recall:** 0.727  
- **F1-Score:** 0.819  

Based on Kaggle’s 40k-review dataset. :contentReference[oaicite:1]{index=1}

---

##  3. Model Architecture (Summary)

1. **Input Preprocessing**  
   - Lowercasing, stopword removal, lemmatization

2. **Dual Feature Extraction**  
   - **RoBERTa embeddings** (768-dim)
   - **Emotion vector** using NRC Lexicon (8-dim)

3. **Feature Fusion**  

   The combined feature representation is constructed by fusing the semantic embedding 
   𝑅  with the emotion feature vector  𝐸, resulting in an integrated feature vector 𝐹.
  

4. **Classification Layer**  
   Fully connected layer + sigmoid

5. **Explainability Layer**  
   SHAP for token-level attributions

---

##  4. Results Summary

### ✔ Performance Metrics  
| Metric | Score |
|--------|--------|
| Accuracy | **0.844** |
| Precision | **0.936** |
| Recall | **0.727** |
| F1-Score | **0.819** |

### ✔ Ablation Study  
| Model Variant | F1-Score |
|---------------|----------|
| RoBERTa only | 0.781 |
| Emotion Only | 0.715 |
| **RoBERTa + Emotion Fusion** | **0.819** |

### ✔ Confusion Matrix Insight  
- High true positives for fake reviews  
- Low false positives  
- Misclassifications reduced because of emotional cues  

---

##  5. Folder Structure (Recommended)
```
Fake-Review-Detection/
│
├── notebooks/
│   └── model_code.ipynb
│
├── data/
│   └── fake_reviews_dataset.csv
│
├── docs/
│   └── Fake Review Detection Thesis.docx
│
└── README.md

```

---

##  6. How to Run the Project

### **1. Install Dependencies**
```bash
pip install torch transformers pandas numpy scikit-learn shap nrclex matplotlib seaborn
```

### **2. Load and Preprocess Dataset**
```python
from nrclex import NRCLex
```

### **3. Train the Model**
```python
python model_code.ipynb
```

---

##  7. Conclusion

This project demonstrates that combining **transformer-based contextual learning** with **emotion features** creates a more reliable and interpretable fake review detector.  
The use of SHAP adds transparency, making it suitable for research, auditing, and deployment in e-commerce review systems.  
Future enhancements may include reviewer-level metadata, multilingual extensions, and a real-time API. 

---

##  8. Authors

- **Shaleni M** – Data preprocessing, emotion feature extraction, implementation  
- **Sandra Sebastian** – Model training, explainability, pipeline integration  
- **Keerthiga M** – Evaluation, visualization, documentation  

Affiliation: School of Advanced Sciences, VIT Chennai.

---


