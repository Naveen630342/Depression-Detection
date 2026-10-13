# Depression Detection Using Machine Learning
## Reddit Dataset Analysis & Text Classification

A machine learning project that analyzes cleaned Reddit text and classifies posts into two dataset labels: **depression (`1`)** and **not depression (`0`)**. The notebook demonstrates text preprocessing, TF-IDF feature extraction, model training, evaluation, and prediction on user-entered text.

> **Important:** This is an educational text-classification project, not a clinical diagnostic tool. Its predictions must not be used to diagnose depression or replace assessment by a qualified mental health professional.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Workflow](#workflow)
- [Models and Evaluation](#models-and-evaluation)
- [Example Prediction](#example-prediction)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Disclaimer](#disclaimer)

## Project Overview

Online text can contain language associated with a person's mood or experiences. This project applies natural language processing (NLP) and traditional machine learning to Reddit text and predicts the label provided by the dataset.

The notebook compares three classifiers:
- Logistic Regression
- Random Forest
- Multinomial Naive Bayes

It reports accuracy, precision, recall, F1-score, a classification report, and a confusion matrix.

## Features

- Loads and inspects a labeled Reddit text dataset.
- Removes duplicate text entries.
- Cleans text by lowercasing and removing URLs, HTML tags, punctuation, digits, and extra whitespace.
- Tokenizes text with NLTK.
- Removes English stopwords and applies WordNet lemmatization.
- Converts text into numerical features using TF-IDF.
- Splits data into training and testing sets with stratification.
- Demonstrates SMOTE resampling on the training features.
- Trains and evaluates multiple classification models.
- Accepts user-entered text for a sample prediction.

## Technologies Used

- Python
- Pandas
- Regular expressions (`re`) and Python `string`
- NLTK
- scikit-learn
- imbalanced-learn (SMOTE)
- Jupyter Notebook

## Dataset

The notebook expects a CSV file named:

```text
depression_dataset_reddit_cleaned.csv
```

The dataset must contain at least these columns:

| Column | Description |
|---|---|
| `clean_text` | Reddit text used as the model input |
| `is_depression` | Binary target label (`0` or `1`) |

Place the CSV in the notebook's working directory, or update the path in the notebook to match its location. Make sure you have permission to use and share the dataset; this repository's code does not automatically include the CSV.

## Project Structure

```text
Depression-Detection/
├── ML_Project.ipynb
├── depression_dataset_reddit_cleaned.csv  # add separately if permitted
└── README.md
```

The CSV is shown as an expected input; it may not be included in the repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Naveen630342/Depression-Detection.git
cd Depression-Detection
```

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas nltk scikit-learn imbalanced-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `ML_Project.ipynb` and run the notebook cells in order.

The notebook downloads NLTK resources for tokenization, stopwords, and WordNet. If your NLTK version requires it, the notebook also downloads `punkt_tab`.

## Workflow

1. **Load and inspect data:** Read the CSV, inspect its dimensions and missing values, and examine the text column.
2. **Remove duplicates:** Drop duplicate rows based on `clean_text`.
3. **Clean text:** Lowercase text and remove URLs, HTML tags, punctuation, digits, non-letter characters, and repeated whitespace.
4. **Tokenize:** Split cleaned text into tokens using NLTK.
5. **Remove stopwords:** Filter common English stopwords.
6. **Lemmatize:** Reduce words to their base form using `WordNetLemmatizer`.
7. **Split data:** Create a 70% training and 30% test split with `random_state=42` and stratification by target label.
8. **Extract features:** Fit `TfidfVectorizer(max_features=5000)` on the training text and transform the test text.
9. **Train models:** Fit Logistic Regression, Random Forest, and Multinomial Naive Bayes.
10. **Evaluate:** Compare classification metrics and inspect the Logistic Regression classification report and confusion matrix.
11. **Predict:** Apply the same text-cleaning steps to user input and generate a prediction using the trained Logistic Regression model.

## Models and Evaluation

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline text classifier |
| Random Forest | Ensemble classifier |
| Multinomial Naive Bayes | Probabilistic classifier commonly used for text classification |

The notebook prints these metrics for each model:

- **Accuracy:** Fraction of predictions that are correct.
- **Precision:** Fraction of positive predictions that are correct.
- **Recall:** Fraction of actual positive examples that are identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts of correct and incorrect predictions by class.

The notebook also creates SMOTE-resampled training data, but the current model-fitting code trains on `X_train_tfidf` and `y_train`, not `X_train_smote` and `y_train_smote`. Therefore, the current model results are **not SMOTE-trained results**. If resampling is intended, train the models using the resampled training data and compare performance on the untouched test set.

## Example Prediction

When the final notebook cell runs, it prompts:

```text
Enter text to check:
```

After entering text, it prints one of the dataset-label predictions:

```text
→ Prediction: DEPRESSED
```

or

```text
→ Prediction: NOT DEPRESSED
```

These outputs reflect the model's learned dataset labels only. They do not establish a person's mental health status.

## Limitations and Future Improvements

- Dataset labels and Reddit language may be noisy, biased, or unrepresentative.
- Text alone cannot reliably establish whether someone has depression.
- Accuracy alone is insufficient; review precision, recall, F1-score, per-class metrics, and confusion matrices.
- Consider cross-validation and hyperparameter tuning.
- Consider comparing models with and without SMOTE, using resampling only on training data.
- Check for data leakage, duplicate or near-duplicate posts across splits, and class imbalance.
- Save the fitted vectorizer and model with an appropriate model-persistence tool if you want to deploy predictions.
- Add dependency versions, tests, and a documented prediction interface for reproducibility.
- Use privacy-respecting, appropriately licensed data and avoid exposing personal information.

## Disclaimer

This project is intended for learning and experimentation with NLP and machine learning. It is **not a medical device or a substitute for professional diagnosis, treatment, or advice**. Do not use its predictions to make decisions about a person's mental health.

## Author

**Naveen Reddy Sagam**

GitHub: [Naveen630342](https://github.com/Naveen630342)
