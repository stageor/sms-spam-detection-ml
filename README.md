# 📧 Email Spam Classifier using Naive Bayes

A machine learning project that classifies Email messages as **spam** or **ham (legitimate)** using the Multinomial Naive Bayes algorithm. The pipeline covers everything from raw text preprocessing to model evaluation, with an optional TF-IDF feature enhancement layer.

---

## 📌 Project Overview

Unsolicited messages are one of the oldest problems in digital communication. This project builds a lightweight, interpretable text classifier trained on a real-world Email dataset. The goal was to understand how probabilistic models handle natural language — and to do it from scratch without hiding the mechanics behind high-level abstractions.

---

## 📂 Dataset

- **Source:** Email Spam Collection Dataset
- **Size:** 5,574 English Email messages
- **Format:** Two columns — `spam` (label: 0 = ham, 1 = spam) and `text` (raw message content)
- **Class Distribution:**
  - ~87% Ham (legitimate messages)
  - ~13% Spam

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data loading and manipulation |
| Matplotlib & Seaborn | Exploratory visualization |
| NLTK | Stopword removal |
| Scikit-learn | Vectorization, model training, evaluation |

---

## 🔄 Project Pipeline

### Step 0 — Import Libraries
Standard data science stack: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

### Step 1 — Load the Dataset
The dataset is loaded from a CSV file and inspected using `.head()`, `.tail()`, `.describe()`, and `.info()` to understand its structure.

### Step 2 — Exploratory Data Analysis
- Computed **message lengths** and visualized their distribution using histograms
- Split messages into spam and ham subsets for comparative analysis
- Inspected the longest message to check for data anomalies
- Plotted class distribution using Seaborn's `countplot`

### Step 3 — Text Preprocessing

**3.1 — Punctuation Removal**
Stripped all punctuation characters from messages using Python's built-in `string.punctuation`.

**3.2 — Stopword Removal**
Downloaded and applied NLTK's English stopwords list to remove common words (e.g., *"the"*, *"is"*, *"and"*) that carry little predictive signal.

**3.3 — CountVectorizer**
Demonstrated how `CountVectorizer` works on a small sample before applying it to the full dataset. The custom `message_cleaning` function (combining punctuation and stopword removal) was plugged directly into the vectorizer's `analyzer` parameter.

### Step 4 — Model Training

- Applied `CountVectorizer` with the custom preprocessing pipeline to transform raw messages into a document-term matrix
- Trained a **Multinomial Naive Bayes** classifier on the full dataset initially, then repeated with a proper **80/20 train-test split**
- Tested the model on custom messages like `"Free money!!!"` and hotel booking inquiries to verify real-world intuition

### Step 5 — Model Evaluation

- Generated **confusion matrices** for both training and test sets, visualized as heatmaps
- Printed a full **classification report** (precision, recall, F1-score) on the test set

### Step 6 — TF-IDF Enhancement

Replaced raw word counts with **TF-IDF (Term Frequency–Inverse Document Frequency)** weights using `TfidfTransformer`. This accounts for the fact that words appearing frequently across all documents (not just spam) should carry less weight. The model was retrained and re-evaluated under this representation.

---

## 📊 Results

The Multinomial Naive Bayes classifier achieved strong performance on the test set, particularly for spam detection. TF-IDF weighting was explored as an alternative feature representation to raw counts.

> Confusion matrix heatmaps and the full classification report are available inside the notebook.

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/spam-classifier-naive-bayes.git
cd spam-classifier-naive-bayes
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

**3. Download NLTK stopwords (first run only)**
```python
import nltk
nltk.download('stopwords')
```

**4. Place the dataset**

Ensure `emails.csv` is in the same directory as the notebook. The file should contain two columns: `text` and `spam`.

**5. Launch the notebook**
```bash
jupyter notebook Spam_Classifier_using_Naive_Bayes.ipynb
```

---

## 💡 Key Concepts Covered

- **Bag of Words** representation with `CountVectorizer`
- **Text normalization**: punctuation removal + stopword filtering
- **Multinomial Naive Bayes**: how it works and why it suits text classification
- **TF-IDF**: intuition and implementation
- **Model evaluation**: confusion matrix, precision, recall, F1-score

---

## 🧠 Reflections

Building this classifier reinforced how much power lies in thoughtful preprocessing. The Naive Bayes model — despite its simplifying independence assumption — performs remarkably well on text data, especially after stripping noise like punctuation and stopwords. The TF-IDF extension showed how feature representation choices directly affect what the model "sees," independent of the algorithm itself.

---

## 📁 Repository Structure

```
spam-classifier-naive-bayes/
│
├── Spam_Classifier_using_Naive_Bayes.ipynb   # Main notebook
├── emails.csv                                 # Dataset (not included, add your own)
└── README.md                                  # Project documentation
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
