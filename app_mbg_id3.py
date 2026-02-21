import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree  # untuk konstanta TREE_UNDEFINED

# =========================
# GLOBALS
# =========================
dataset = None
model = None
label_encoder = None
feature_names = None

# Nama fitur yang akan dipakai (biner) — sesuai analisis ID3
BINARY_FEATURES = [
    "ada_data_bukti",
    "sumber_resmi",
    "kata_provokatif",
    "klaim_berlebihan"
]

# Untuk menampilkan nama fitur yang lebih manusiawi pada output pohon
HUMAN_FNAMES = {
    "ada_data_bukti": "Ada data atau bukti",
    "sumber_resmi": "Sumber resmi",
    "kata_provokatif": "Bahasa provokatif",
    "klaim_berlebihan": "Klaim berlebihan"
}

# =========================
# FEATURE EXTRACTION (from title text)
# =========================
def extract_features(text):
    """
    Kembalikan dict fitur biner (0/1) berdasarkan teks judul_post.
    Dipakai saat dataset tidak menyediakan kolom atribut eksplisit.
    """
    if not isinstance(text, str):
        text = ""

    text_lower = text.lower()

    # kata / frasa yang menunjukkan ada bukti atau data
    bukti_phrases = [
        "berdasarkan data", "hasil penelitian", "laporan resmi",
        "menurut data", "statistik", "hasil survei", "ada bukti",
        "menunjukkan", "ditemukan", "tertulis", "rilis resmi", "menurut laporan"
    ]

    # indikasi sumber resmi (domain / nama instansi)
    sumber_patterns = r"\.go\.id|\.ac\.id|kemenkes|kemendikbud|kominfo|bps|kemensos|kemendagri|kemendesa"

    # kata yang terkesan provokatif / emosional
    provokatif_words = [
        "viral", "heboh", "menghebohkan", "skandal", "mengancam",
        "bahaya", "mengguncang", "rusak", "mencekam",
        "mengejutkan", "bohong"
    ]

    # klaim yang terkesan berlebihan / absolut
    klaim_words = [
        "pasti", "100%", "tanpa bukti", "tidak diragukan", "selalu",
        "jaminan", "tidak salah", "terbukti", "pasti akan"
    ]

    ada_bukti = int(any(p in text_lower for p in bukti_phrases) or bool(re.search(r"\bdata\b|\bbukti\b|\blaporan\b|\bsurvei\b", text_lower)))
    sumber_resmi = int(bool(re.search(sumber_patterns, text_lower)))
    kata_provokatif = int(any(w in text_lower for w in provokatif_words))
    klaim_berlebihan = int(any(w in text_lower for w in klaim_words))

    return {
        "ada_data_bukti": ada_bukti,
        "sumber_resmi": sumber_resmi,
        "kata_provokatif": kata_provokatif,
        "klaim_berlebihan": klaim_berlebihan
    }

# =========================
# LOAD DATASET (CSV or XLSX)
# =========================
def load_dataset():
    global dataset
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
    )
    if not file_path:
        return

    try:
        if file_path.lower().endswith(".csv"):
            dataset = pd.read_csv(file_path)
        elif file_path.lower().endswith(".xlsx"):
            dataset = pd.read_excel(file_path)
        else:
            messagebox.showerror("Error", "Format file tidak didukung. Gunakan .csv atau .xlsx")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Gagal membaca file:\n{e}")
        dataset = None
        return

    if "judul_post" not in dataset.columns and not set(BINARY_FEATURES).issubset(dataset.columns):
        messagebox.showerror("Error", "Dataset harus memiliki kolom 'judul_post' atau kolom atribut eksplisit: " + ", ".join(BINARY_FEATURES))
        dataset = None
        return

    if "label" not in dataset.columns:
        messagebox.showerror("Error", "Dataset harus memiliki kolom 'label'")
        dataset = None
        return

    messagebox.showinfo("Info", "Dataset berhasil dimuat")

# =========================
# UTILS: normalize yes/no -> 1/0 (if dataset contains 'ya'/'tidak')
# =========================
def normalize_boolean_like_columns(df, cols):
    """
    Jika kolom berisi 'ya'/'tidak' atau 'yes'/'no', map ke 1/0.
    Jika berisi 1/0 sudah, biarkan.
    Jika ada non-convertible values -> NaN (cek nanti).
    """
    mapping = {
        "ya": 1, "tidak": 0, "yes": 1, "no": 0,
        "true": 1, "false": 0, "1": 1, "0": 0
    }

    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            series = df2[c]
            if series.dtype == object:
                mapped = series.astype(str).str.strip().str.lower().map(mapping)
                df2[c] = mapped
            else:
                try:
                    df2[c] = series.astype(int)
                except Exception:
                    df2[c] = series.astype(str).str.strip().str.lower().map(mapping)
    return df2

# =========================
# HELPERS: readable tree & feature importances
# =========================
def build_readable_tree(clf, feature_names, class_names, prefer_yes_first=True):
    """
    Menghasilkan string tree yang mudah dibaca, mis:
    ROOT: Ada data atau bukti?
    ├── Ya
    │   └── LEAF: Fakta
    ...
    - clf: DecisionTreeClassifier terlatih
    - feature_names: list nama fitur yang dipakai (urutan sama seperti training)
    - class_names: array label asli (LabelEncoder.classes_)
    - prefer_yes_first: jika True, tampilkan cabang 'Ya' (value 1) dulu untuk fitur biner
    """
    tree = clf.tree_
    indent_unit = "    "

    def is_binary_feature(node):
        # heuristik: threshold sekitar 0.5 menandakan binary 0/1 split
        thr = tree.threshold[node]
        return thr <= 0.5 + 1e-9 and thr >= 0.5 - 1e-9

    def node_label_from_class(node):
        vals = tree.value[node][0]
        idx = int(np.argmax(vals))
        return class_names[idx].title()

    def recurse(node, depth=0):
        indent = indent_unit * depth
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            # leaf
            label = node_label_from_class(node)
            return f"{indent}LEAF: {label}\n"
        # internal node
        fname = feature_names[tree.feature[node]]
        human_name = HUMAN_FNAMES.get(fname, fname.replace("_", " ").title())
        out = f"{indent}{human_name}?\n"

        left = tree.children_left[node]
        right = tree.children_right[node]

        if is_binary_feature(node) and prefer_yes_first:
            # print 'Ya' (right child) first for readability
            out += f"{indent}├── Ya\n"
            out += recurse(right, depth + 1)
            out += f"{indent}└── Tidak\n"
            out += recurse(left, depth + 1)
        else:
            thr = tree.threshold[node]
            out += f"{indent}├── {human_name} <= {thr:.2f}\n"
            out += recurse(left, depth + 1)
            out += f"{indent}└── {human_name} > {thr:.2f}\n"
            out += recurse(right, depth + 1)
        return out

    # Build top line (root)
    if tree.feature[0] != _tree.TREE_UNDEFINED:
        root_name = feature_names[tree.feature[0]]
        root_human = HUMAN_FNAMES.get(root_name, root_name.replace("_", " ").title())
        tree_str = f"ROOT: {root_human}?\n\n"
    else:
        tree_str = "ROOT: (leaf)\n\n"
    tree_str += recurse(0, depth=0)
    return tree_str

def format_feature_importances(clf, feature_names):
    imps = clf.feature_importances_
    if imps is None or float(imps.sum()) == 0.0:
        return "Feature importances: not available\n"
    pairs = list(zip(feature_names, imps))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    lines = []
    for name, imp in pairs_sorted:
        human = HUMAN_FNAMES.get(name, name.replace("_", " ").title())
        pct = round(float(imp) * 100, 1)
        lines.append(f"- {human} ({pct}%)")
    return "Feature Importances:\n" + "\n".join(lines) + "\n"

# =========================
# TRAIN MODEL (extract features from title if needed)
# =========================
def train_model():
    global model, label_encoder, feature_names, dataset

    if dataset is None:
        messagebox.showwarning("Warning", "Load dataset dulu")
        return

    df = dataset.copy()

    # If explicit attribute columns exist, prefer them.
    if set(BINARY_FEATURES).issubset(df.columns):
        # normalize possible 'ya'/'tidak' values to 0/1
        df = normalize_boolean_like_columns(df, BINARY_FEATURES)
        try:
            X = df[BINARY_FEATURES].astype(int)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengkonversi atribut menjadi numerik:\n{e}")
            return
    else:
        # Need to extract features from judul_post
        if "judul_post" not in df.columns:
            messagebox.showerror("Error", "Tidak ada kolom atribut dan juga tidak ada 'judul_post' untuk ekstraksi fitur.")
            return

        rows = []
        for t in df["judul_post"].astype(str):
            rows.append(extract_features(t))
        X = pd.DataFrame(rows, dtype=int)

    y = df["label"].astype(str)

    # store feature names
    feature_names = X.columns.tolist()

    # encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # train Decision Tree (ID3 style: entropy)
    model = DecisionTreeClassifier(criterion="entropy", random_state=42, min_samples_leaf=2)
    model.fit(X, y_encoded)

    # Build readable tree and formatted feature importances
    readable = build_readable_tree(model, feature_names, label_encoder.classes_, prefer_yes_first=True)
    fi_text = format_feature_importances(model, feature_names)

    # tree_text.delete("1.0", tk.END)
    # tree_text.insert(tk.END, "=== Decision Tree (versi ramah pengguna) ===\n\n")
    # tree_text.insert(tk.END, readable)
    tree_text.insert(tk.END, "\n=== Interpretasi Feature Importances ===\n")
    tree_text.insert(tk.END, fi_text)
    # Nilai ini menunjukkan seberapa besar setiap atribut memengaruhi keputusan model dalam membedakan berita fakta dan hoaks. 
    # Semakin besar persentasenya, semakin sering atribut tersebut digunakan dalam proses pengambilan keputusan.”

    messagebox.showinfo("Info", "Model berhasil dilatih (fitur: " + ", ".join(feature_names) + ")")

# =========================
# PREDICT (extract features from input text)
# =========================
def predict():
    global model, label_encoder, feature_names
    if model is None:
        messagebox.showwarning("Warning", "Model belum dilatih")
        return

    text = input_entry.get().strip()
    if not text:
        messagebox.showwarning("Warning", "Masukkan teks postingan")
        return

    # Build input features: if model trained on explicit columns, still we extract from text
    feats = extract_features(text)
    X_input = pd.DataFrame([feats], dtype=int)

    # ensure column order matches
    X_input = X_input[feature_names]

    result_index = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]
    confidence = probs[result_index]
    label = label_encoder.inverse_transform([result_index])[0]

    result_label.config(text=f"Hasil Prediksi: {label}  (confidence: {confidence*100:.1f}%)")

    # show extracted features
    explanation_text.delete("1.0", tk.END)
    explanation_text.insert(tk.END, "Fitur hasil ekstraksi dari judul:\n")
    for k in feature_names:
        explanation_text.insert(tk.END, f"- {HUMAN_FNAMES.get(k,k)}: {int(X_input.iloc[0][k])}\n")

    # optional context check (MBG)
    if "mbg" not in text.lower() and "makan bergizi" not in text.lower():
        # hanya peringatan, bukan error
        messagebox.showwarning("Peringatan", "Teks tidak mengandung kata kunci MBG (makan bergizi). Hasil mungkin tidak relevan.")

# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Klasifikasi Hoaks MBG (judul_post → fitur) - ID3")
root.geometry("920x700")

top = tk.Frame(root)
top.pack(pady=8)

btn_load = tk.Button(top, text="Load Dataset (.csv/.xlsx)", command=load_dataset)
btn_load.pack(side=tk.LEFT, padx=6)

btn_train = tk.Button(top, text="Train Model (ID3)", command=train_model)
btn_train.pack(side=tk.LEFT, padx=6)

# input
tk.Label(root, text="Masukkan Judul Postingan:", font=("Arial", 11)).pack(anchor="w", padx=10, pady=(10,0))
input_entry = tk.Entry(root, width=120)
input_entry.pack(padx=10, pady=6)

btn_predict = tk.Button(root, text="Prediksi", bg="#2c7be5", fg="white", command=predict)
btn_predict.pack(pady=6)

result_label = tk.Label(root, text="Hasil Prediksi: -", font=("Arial", 12, "bold"))
result_label.pack(pady=6)

tk.Label(root, text="Fitur Ekstraksi / Keterangan:", font=("Arial", 10, "bold")).pack(anchor="w", padx=10)
explanation_text = tk.Text(root, height=6)
explanation_text.pack(fill=tk.X, padx=10, pady=6)

tk.Label(root, text="Kontribusi Atribut dalam Keputusan Model", font=("Arial", 10, "bold")).pack(anchor="w", padx=10)
tree_text = tk.Text(root, height=18)
tree_text.pack(fill=tk.BOTH, padx=10, pady=6, expand=True)

root.mainloop()
