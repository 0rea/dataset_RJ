import joblib
from pythainlp.tokenize import word_tokenize

# ต้องมีฟังก์ชันนี้เหมือนใน train.py
def tokenize(text):
    return word_tokenize(text)

print("Loading AI model...")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("ระบบตรวจคำหยาบพร้อมใช้งาน\n")

while True:

    msg = input("พิมพ์ข้อความ: ")

    msg_vec = vectorizer.transform([msg])

    pred = model.predict(msg_vec)

    if pred[0] == 1:
        print("⚠️ พบคำหยาบ\n")
    else:
        print("✅ ข้อความปกติ\n")