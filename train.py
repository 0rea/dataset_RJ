import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

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
model = SVC(kernel="linear")
model.fit(X_train, y_train)


# ทดสอบโมเดล
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix\n")
cm = confusion_matrix(y_test, pred)
print(cm)


# ===== Confusion Matrix Graph =====
ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.title("Confusion Matrix")
plt.show()


# ===== Hyperplane Visualization =====
# ลดมิติด้วย PCA
pca = PCA(n_components=2)

X_2d = pca.fit_transform(X_vector.toarray())

svm_2d = SVC(kernel="linear")
svm_2d.fit(X_2d, y)

plt.figure()

plt.scatter(
    X_2d[:,0],
    X_2d[:,1],
    c=y,
    cmap="coolwarm",
    s=20
)

ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = svm_2d.decision_function(xy)
Z = Z.reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[0], colors="black")

plt.title("SVM Hyperplane Visualization")
plt.show()


# save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved!")