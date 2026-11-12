# Mental Health Text Classification Using Machine Learning

## Overview
A Natural Language Processing (NLP) and Machine Learning project that analyzes textual data to classify mental health-related content into different categories. The project explores multiple machine learning algorithms and TF-IDF feature extraction to identify patterns in text and evaluate classification performance.

## Objectives
- Process and clean textual data using NLP techniques.
- Extract meaningful text features using TF-IDF.
- Train and compare multiple machine learning classification algorithms.
- Evaluate model performance using standard classification metrics.

## Dataset
- **Total Samples:** 5,957
- **Input Features:** Text and Title
- **Target:** Five classification categories (encoded as labels 0–4).
- **Class Distribution:** Approximately balanced across all five categories.

The dataset contains textual posts and titles related to mental health.

## Methodology

### 1. Data Preprocessing
- Handle missing values.
- Combine title and text into a single feature.
- Convert text to lowercase.
- Remove unnecessary special characters.
- Remove English stopwords while preserving important textual information.

### 2. Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical feature vectors.

Different configurations are explored:
- Word-level features (unigrams and bigrams).
- Character-level n-grams.
- Different maximum vocabulary sizes.

### 3. Machine Learning Algorithms
The following algorithms are explored and compared:

- Multinomial Naive Bayes
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- XGBoost
- Multi-Layer Perceptron (MLP)

### 4. Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Experiments are conducted with different algorithms and feature extraction configurations to identify the most effective classification approach.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, NLTK
- **Machine Learning:** Scikit-learn, XGBoost
- **NLP:** TF-IDF Vectorizer
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/CybilFa/Mental-Health-Text-Classification.git
cd Mental-Health-Text-Classification
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib xgboost
```

### 4. Download NLTK Stopwords
```python
import nltk
nltk.download('stopwords')
```

### 5. Run the Project
Open the Jupyter Notebook in VS Code or Jupyter Notebook and execute the cells sequentially to preprocess the data, extract features, train models, and evaluate their performance.

## Project Structure

```text
MentalHealthClassification/
│
├── notebooks/              # Jupyter notebooks and experiments
├── data/                   # Dataset files (if included)
├── models/                 # Saved trained models
├── mental_health_model.pkl # Trained classification model
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

*Note: The structure above is a suggested organization; adjust it to match the actual files in your repository.*

## Future Improvements
- Experiment with pretrained transformer models such as BERT.
- Explore semantic embeddings for improved text representation.
- Perform hyperparameter optimization.
- Improve classification performance through further experimentation.
- Develop an interactive interface for text-based predictions.

## Disclaimer
This project is developed for educational and research purposes. Its predictions are not a substitute for professional mental health assessment, diagnosis, or treatment.

## Author
**Cybil Fatima**

[GitHub](https://github.com/CybilFa)