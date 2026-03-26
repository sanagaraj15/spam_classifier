import pandas as pd
import re
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()
def train_model():
    print("\nLoading dataset...")
    data = pd.read_csv("spam.csv", encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    data['message'] = data['message'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    print("\nModel saved successfully.")
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        print("\nModel not found. Please train the model first.")
        return None, None
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer
def predict():
    model, vectorizer = load_model()
    if model is None:
        return
    print("\nEnter messages (type 'exit' to stop)")
    while True:
        msg = input("\nMessage: ")
        if msg.lower() == "exit":
            break
        msg_clean = preprocess(msg)
        msg_vec = vectorizer.transform([msg_clean])
        result = model.predict(msg_vec)[0]
        if result == 1:
            print("Spam")
        else:
            print("Not Spam")
def main():
    while True:
        print("\n===== Spam Classifier =====")
        print("1. Train Model")
        print("2. Test Message")
        print("3. Exit")
        choice = input("\nEnter choice: ")

        if choice == "1":
            train_model()
        elif choice == "2":
            predict()
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice.")
if __name__ == "__main__":
    main()