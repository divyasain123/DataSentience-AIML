# 🌐 DataSentience-AIML 💡

**Part of Social Summer of Code 2025 🚀 & GirlScript Summer of Code 2025 🚀**

Harnessing the power of **AI & ML** across **Healthcare, Finance, Agriculture, NLP, Cyber-Safety & more.**
Open-source contributions are welcome! 🤝



## 🔥 Project Highlights

* 🤖 Cutting-edge **AI/ML models** applied across multiple domains.
* 🧠 Current Work: **Fake Review Detection** using NLP + ML.
* 📊 Added datasets, preprocessing pipelines & extended research notebooks.
* 🌍 Built under **SSoC '25** & **GSSoC '25** initiatives with open-source spirit.

---

## 🛠️ Tech Stack

* **Languages**: Python 🐍
* **Libraries**: Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn
* **Tools**: Jupyter Notebook, Git, Open Source Frameworks

---

## 📂 Repo Structure

```
DataSentience-AIML/
│
├── Fake-Reviews-Detection/
│   ├── datasets/                  # Raw and processed datasets
│   ├── notebooks/                 # Jupyter notebooks for experiments
│   ├── scripts/                   # Python scripts for preprocessing & modeling
│   ├── models/                    # Saved ML models
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Module-specific README
│
├── Healthcare/
├── Finance/
├── Agriculture/
├── NLP/
├── Cyber-Safety/
│
└── README.md                      # Main repo README
```

---

## 📝 Project: Fake Reviews Detection

### Problem Statement

Detect fake reviews from a massive collection of product reviews across various categories like **Home & Office**, **Sports**, etc.

* **Labels**:

  * **OR** → Original reviews (human-generated)
  * **CG** → Computer-generated fake reviews
* **Objective**: Identify if a review is **fraudulent** (fake) or genuine.

---

### Dataset Description

* Contains **20k fake reviews** (CG) and **20k real reviews** (OR).
* Each review includes:

  * Text content
  * Rating
  * Label (OR/CG)

---

### Python Libraries & Packages Used

```text
numpy, pandas, matplotlib.pyplot, seaborn, warnings,
nltk, string, sklearn.naive_bayes, sklearn.feature_extraction,
sklearn.model_selection, sklearn.ensemble, sklearn.tree,
sklearn.linear_model, sklearn.svm, sklearn.neighbors
```

---

### Text Preprocessing Techniques

* Removing punctuation
* Converting text to lowercase
* Eliminating stopwords
* Stemming & Lemmatizing
* Removing digits

---

### Transformers for Text Vectorization

* **CountVectorizer** (Bag-of-Words)
* **TF-IDF** (Term Frequency-Inverse Document Frequency)

---

### Machine Learning Algorithms Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Classifier (SVC)
* Decision Tree Classifier
* Random Forest Classifier
* Multinomial Naive Bayes

---

### Performance Overview

| Algorithm               | Accuracy |
| ----------------------- | -------- |
| SVC                     | 88%      |
| Logistic Regression     | 86%      |
| Random Forest           | 84%      |
| Multinomial Naive Bayes | 84%      |
| Decision Tree           | 73%      |
| KNN                     | 58%      |

**Insight**: SVC performed best, making it the most reliable choice for fake review detection. KNN was the least accurate.

---
