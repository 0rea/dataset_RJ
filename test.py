import joblib
import matplotlib.pyplot as plt
import numpy as np

from pythainlp.tokenize import word_tokenize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# ฟังก์ชัน tokenize ต้องเหมือน train.py
def tokenize(text):
    return word_tokenize(text)

print("Loading AI model...")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("ระบบตรวจคำหยาบพร้อมใช้งาน\n")

X_data = []
y_data = []

while True:

    msg = input("พิมพ์ข้อความ (พิมพ์ exit เพื่อดูกราฟ): ")

    if msg == "exit":
        break

    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)

    X_data.append(msg_vec.toarray()[0])
    y_data.append(pred[0])

    if pred[0] == 1:
        print("⚠️ พบคำหยาบ\n")
    else:
        print("✅ ข้อความปกติ\n")
