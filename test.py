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


# ===== สร้างกราฟหลังจากออกจาก loop =====

if len(X_data) > 1:

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # ลดมิติให้เหลือ 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_data)

    # train SVM 2D เพื่อวาด hyperplane
    svm_2d = SVC(kernel="linear")
    svm_2d.fit(X_2d, y_data)

    plt.figure()

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_data,
        cmap="coolwarm",
        s=50
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

else:
    print("ข้อมูลไม่พอสำหรับสร้างกราฟ")