"""
Complaint Classification Dashboard
================================

This Streamlit application offers a user‑friendly interface for
classifying citizen complaints into categories relevant to local
government administration.  It uses a Multinomial Naive Bayes model
trained on a synthetic dataset of complaints (`complaints_data.csv`).

Features:

* **Complaint classifier** – Enter a complaint in natural language and
  receive an immediate prediction of which category (Road, Water,
  Electricity, Waste, Health or Other) it belongs to.
* **Dataset summary** – View the number of complaints in each
  category presented in a table.
* **Sample complaints** – Explore example complaints for each
  category to understand typical issues.

To run the dashboard, install the required packages and execute:

```bash
streamlit run complaint_dashboard.py
```

Dependencies:

* streamlit
* pandas
* scikit‑learn

"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(data: pd.DataFrame) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        data['complaint'], data['category'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, classifier.predict(X_test_vec))
    return classifier, vectorizer, acc


def classify(text: str, classifier: MultinomialNB, vectorizer: CountVectorizer) -> str:
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]


def main() -> None:
    st.set_page_config(page_title="Complaint Classification Dashboard", layout="wide")
    st.title("Complaint Classification Dashboard")
    # Load data
    try:
        data = load_data('complaints_data.csv')
    except FileNotFoundError:
        st.error("Dataset 'complaints_data.csv' not found in the working directory.")
        return
    # Train model
    classifier, vectorizer, acc = train_model(data)
    st.write(f"Model accuracy on test set: {acc:.2f}")
    # Input box for classifying a new complaint
    st.subheader("Classify a Complaint")
    complaint_text = st.text_area("Enter complaint description:", height=120)
    if complaint_text:
        category_pred = classify(complaint_text, classifier, vectorizer)
        st.success(f"Predicted Category: {category_pred}")
    # Dataset summary table
    st.subheader("Dataset Summary")
    counts = data['category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    st.table(counts)
    # Sample complaints by category
    st.subheader("Sample Complaints by Category")
    categories = sorted(data['category'].unique())
    tabs = st.tabs(categories)
    for tab, cat in zip(tabs, categories):
        with tab:
            examples = data[data['category'] == cat]['complaint'].head(3).tolist()
            for example in examples:
                st.write(f"- {example}")


if __name__ == '__main__':
    main()