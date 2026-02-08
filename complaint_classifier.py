"""
Complaint Classification
=======================

This script implements a simple text classifier to categorize citizen
complaints submitted to local government administration.  It uses
a Multinomial Naive Bayes model trained on the synthetic
`complaints_data.csv` dataset, which contains examples of complaints
labelled with categories such as Road, Water, Electricity, Waste,
Health and Other.

The script performs the following steps:

1. Load the dataset of complaints and labels.
2. Split the data into training and test sets.
3. Convert text to a bag‑of‑words representation using
   `CountVectorizer`.
4. Train a `MultinomialNB` classifier.
5. Evaluate the classifier and print the accuracy.
6. Provide a command‑line interface for classifying new complaints.

Usage:

```bash
python complaint_classifier.py
```

You can then type a complaint when prompted to see which category
the model predicts.

Dependencies:

* pandas
* scikit‑learn

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        data['complaint'], data['category'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return classifier, vectorizer, acc, report


def classify_complaint(complaint: str, classifier: MultinomialNB, vectorizer: CountVectorizer) -> str:
    vec = vectorizer.transform([complaint])
    return classifier.predict(vec)[0]


def main() -> None:
    try:
        data = load_data('complaints_data.csv')
    except FileNotFoundError:
        print("Error: 'complaints_data.csv' not found.")
        return
    classifier, vectorizer, acc, report = train_model(data)
    print(f"Model trained. Test set accuracy: {acc:.2f}\n")
    print("Classification report:\n")
    print(report)
    print("Enter complaint descriptions to classify them (type 'quit' to exit).\n")
    while True:
        text = input("Complaint: ").strip()
        if text.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
        predicted = classify_complaint(text, classifier, vectorizer)
        print(f"Predicted category: {predicted}\n")


if __name__ == '__main__':
    main()