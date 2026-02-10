import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text

# =========================
# GLOBAL VARIABLE
# =========================
dataset = None
model = None
label_encoder = None
feature_names = None

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(text):
    if not isinstance(text, str):
        text = ""

    text_lower = text.lower()

    provokatif_words = ["bahaya", "bohong", "penipuan", "hancur", "hoaks"]
    klaim_words = ["pasti", "100%", "tanpa bukti", "tidak diragukan"]

    features = {
        "provokatif": int(any(w in text_lower for w in provokatif_words)),
        "klaim_berlebihan": int(any(w in text_lower for w in klaim_words)),
        "sumber_resmi": int(bool(re.search(r"\.go\.id|\.ac\.id|kemenkes|kemendikbud|kominfo|bps", text_lower))),
        "panjang_teks": len(text.split()),
        "huruf_kapital": sum(1 for c in text if c.isupper())
    }

    return features

# =========================
# LOAD DATASET
# =========================
def load_dataset():
    global dataset

    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_path:
        return

    dataset = pd.read_csv(file_path)

    if "judul_post" not in dataset.columns or "label" not in dataset.columns:
        messagebox.showerror(
            "Error",
            "Dataset harus memiliki kolom 'judul_post' dan 'label'"
        )
        dataset = None
        return

    messagebox.showinfo("Info", "Dataset berhasil dimuat")

# =========================
# TRAIN MODEL ID3
# =========================
def train_model():
    global model, label_encoder, feature_names

    if dataset is None:
        messagebox.showwarning("Warning", "Load dataset dulu")
        return

    feature_rows = []

    for text in dataset["judul_post"]:
        feature_rows.append(extract_features(text))

    X = pd.DataFrame(feature_rows)

    # Pastikan semua fitur numerik
    X = X.astype(int)
    feature_names = X.columns.tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset["label"].astype(str))

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    tree_rules = export_text(model, feature_names=feature_names)
    tree_text.delete("1.0", tk.END)
    tree_text.insert(tk.END, tree_rules)

    messagebox.showinfo("Info", "Model ID3 berhasil dilatih")

# =========================
# PREDICTION
# =========================
def predict():
    if model is None:
        messagebox.showwarning("Warning", "Model belum dilatih")
        return

    text = input_entry.get().strip()
    if not text:
        messagebox.showwarning("Warning", "Masukkan teks postingan")
        return

    features = extract_features(text)
    X_input = pd.DataFrame([features])

    # Samakan struktur dengan data training
    X_input = X_input[feature_names]
    X_input = X_input.astype(int)

    result = model.predict(X_input)[0]
    label = label_encoder.inverse_transform([result])[0]

    result_label.config(text=f"Hasil Prediksi: {label}")

    explanation = "Fitur hasil ekstraksi:\n"
    for k, v in features.items():
        explanation += f"- {k}: {v}\n"

    explanation_text.delete("1.0", tk.END)
    explanation_text.insert(tk.END, explanation)

    if "mbg" not in text.lower() and "makan bergizi" not in text.lower():
        messagebox.showwarning(
            "Peringatan",
            "Teks tidak mengandung konteks Program Makan Bergizi Gratis (MBG)"
        )

# =========================
# GUI TKINTER
# =========================
root = tk.Tk()
root.title("Klasifikasi Hoaks MBG - Decision Tree ID3")
root.geometry("760x760")
root.configure(bg="#f4f6f9")

title_label = tk.Label(
    root,
    text="Aplikasi Klasifikasi Hoaks MBG\nDecision Tree ID3",
    font=("Arial", 16, "bold"),
    bg="#f4f6f9"
)
title_label.pack(pady=15)

btn_frame = tk.Frame(root, bg="#f4f6f9")
btn_frame.pack(pady=5)

tk.Button(
    btn_frame, text="Load Dataset",
    width=18, command=load_dataset
).grid(row=0, column=0, padx=10)

tk.Button(
    btn_frame, text="Train Model ID3",
    width=18, command=train_model
).grid(row=0, column=1, padx=10)

tk.Label(
    root, text="Masukkan Teks Postingan:",
    bg="#f4f6f9", font=("Arial", 11)
).pack(pady=8)

input_entry = tk.Entry(root, width=90)
input_entry.pack(pady=5)

tk.Button(
    root, text="Prediksi",
    width=25, bg="#2c7be5", fg="white",
    command=predict
).pack(pady=12)

result_label = tk.Label(
    root, text="Hasil Prediksi: -",
    font=("Arial", 13, "bold"),
    bg="#f4f6f9"
)
result_label.pack(pady=10)

tk.Label(
    root, text="Fitur yang Diekstraksi:",
    bg="#f4f6f9", font=("Arial", 11, "bold")
).pack()

explanation_text = tk.Text(root, height=7, width=90)
explanation_text.pack(pady=5)

tk.Label(
    root, text="Struktur Decision Tree (ID3):",
    bg="#f4f6f9", font=("Arial", 11, "bold")
).pack(pady=5)

tree_text = tk.Text(root, height=16, width=90)
tree_text.pack(pady=5)

root.mainloop()
