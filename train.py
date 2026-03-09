import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pythainlp.tokenize import word_tokenize


# ฟังก์ชันตัดคำภาษาไทย
def tokenize(text):
    return word_tokenize(text)


# โหลด dataset
data = pd.read_csv("thai_chat_offensive_dataset.csv")

X = data["text"]
y = data["label"]


# แปลงข้อความเป็นตัวเลข
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    token_pattern=None,
    ngram_range=(1,2)
)

X_vector = vectorizer.fit_transform(X)


# แบ่ง train test
X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)


# train model
model = SVC()

model.fit(X_train, y_train)


# ทดสอบโมเดล
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, pred))


# save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved!")