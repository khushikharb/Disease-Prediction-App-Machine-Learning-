import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the datasets
liver = pd.read_csv('datasets/liver.csv')
kidney = pd.read_csv('datasets/kidney.csv')
diabetes = pd.read_csv('datasets/diabetes.csv')
heart = pd.read_csv('datasets/heart.csv')

# Fill missing values
liver.fillna(method='ffill', inplace=True)
kidney.fillna(method='ffill', inplace=True)
diabetes.fillna(method='ffill', inplace=True)
heart.fillna(method='ffill', inplace=True)

# Initialize models and accuracy variables globally
model_liver = model_kidney = model_diabetes = model_heart = None
accuracy_liver = accuracy_kidney = accuracy_diabetes = accuracy_heart = 0

# Train models for each disease
def train_models():
    global model_liver, model_kidney, model_diabetes, model_heart
    global accuracy_liver, accuracy_kidney, accuracy_diabetes, accuracy_heart

    # Liver Disease model
    X_l = liver.drop('Diagnosis', axis=1)
    y_l = liver['Diagnosis']
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2)
    model_liver = RandomForestClassifier()
    model_liver.fit(X_train_l, y_train_l)
    y_pred_l = model_liver.predict(X_test_l)
    accuracy_liver = accuracy_score(y_test_l, y_pred_l)

    # Kidney Disease model
    X_k = kidney.drop('classification', axis=1)
    y_k = kidney['classification']
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_k, y_k, test_size=0.2)
    model_kidney = RandomForestClassifier()
    model_kidney.fit(X_train_k, y_train_k)
    y_pred_k = model_kidney.predict(X_test_k)
    accuracy_kidney = accuracy_score(y_test_k, y_pred_k)

    # Diabetes model
    X_d = diabetes.drop('Outcome', axis=1)
    y_d = diabetes['Outcome']
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2)
    model_diabetes = RandomForestClassifier()
    model_diabetes.fit(X_train_d, y_train_d)
    y_pred_d = model_diabetes.predict(X_test_d)
    accuracy_diabetes = accuracy_score(y_test_d, y_pred_d)

    # Heart Disease model
    X_h = heart.drop('HeartDisease', axis=1)
    y_h = heart['HeartDisease']
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2)
    model_heart = RandomForestClassifier()
    model_heart.fit(X_train_h, y_train_h)
    y_pred_h = model_heart.predict(X_test_h)
    accuracy_heart = accuracy_score(y_test_h, y_pred_h)

# Update input fields based on selected disease
def update_input_fields(event):
    disease = disease_combobox.get()

    for widget in input_frame.winfo_children():
        widget.destroy()
    input_entries.clear()

    if disease == "Liver Disease":
        labels = list(liver.columns[:-1])
        accuracy = accuracy_liver
    elif disease == "Kidney Disease":
        labels = list(kidney.columns[:-1])
        accuracy = accuracy_kidney
    elif disease == "Diabetes":
        labels = list(diabetes.columns[:-1])
        accuracy = accuracy_diabetes
    elif disease == "Heart Disease":
        labels = list(heart.columns[:-1])
        accuracy = accuracy_heart
    else:
        labels = []
        accuracy = None

    create_input_fields(labels)

    if accuracy is not None:
        accuracy_label.config(text=f"Model Accuracy: {accuracy * 100:.2f}%")
    else:
        accuracy_label.config(text="")

# Create input fields dynamically
def create_input_fields(labels):
    for idx, label in enumerate(labels):
        lbl = tk.Label(input_frame, text=label, anchor="w", font=("Arial", 10))
        lbl.grid(row=idx, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(input_frame, font=("Arial", 10))
        entry.grid(row=idx, column=1, padx=5, pady=5, sticky="ew")
        input_entries.append(entry)

    input_frame.grid_columnconfigure(1, weight=1)

# Prediction function
def make_prediction():
    disease = disease_combobox.get()

    try:
        inputs = [float(entry.get()) for entry in input_entries]
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")
        return

    if disease == "Liver Disease":
        features = list(liver.columns[:-1])
        input_df = pd.DataFrame([inputs], columns=features)
        prediction = model_liver.predict(input_df)[0]
        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"

    elif disease == "Kidney Disease":
        features = list(kidney.columns[:-1])
        input_df = pd.DataFrame([inputs], columns=features)
        prediction = model_kidney.predict(input_df)[0]
        result = "Kidney Disease Detected" if prediction == 'ckd' else "No Kidney Disease"

    elif disease == "Diabetes":
        features = list(diabetes.columns[:-1])
        input_df = pd.DataFrame([inputs], columns=features)
        prediction = model_diabetes.predict(input_df)[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"

    elif disease == "Heart Disease":
        features = list(heart.columns[:-1])
        input_df = pd.DataFrame([inputs], columns=features)
        prediction = model_heart.predict(input_df)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    else:
        result = "Please select a disease."

    result_label.config(text=f"Result: {result}", fg="green")

# Initialize main window
root = tk.Tk()
root.title("Multiple Disease Prediction System")
root.geometry("700x600")
root.configure(bg="#f0f4f7")

# Styling
style = ttk.Style()
style.configure('TButton', font=('Arial', 11), padding=5)
style.configure('TLabel', font=('Arial', 11))
style.configure('TCombobox', font=('Arial', 11))

# Title Label
title_label = tk.Label(root, text="Multiple Disease Prediction System", font=("Arial", 16, "bold"), bg="#f0f4f7", fg="#333")
title_label.pack(pady=15)

# Disease Selection
selection_frame = tk.Frame(root, bg="#f0f4f7")
selection_frame.pack(pady=10)

disease_label = tk.Label(selection_frame, text="Select Disease:", font=("Arial", 12), bg="#f0f4f7")
disease_label.pack(side="left", padx=(0, 10))

disease_combobox = ttk.Combobox(selection_frame, values=["Liver Disease", "Kidney Disease", "Diabetes", "Heart Disease"], state="readonly")
disease_combobox.pack(side="left")
disease_combobox.bind("<<ComboboxSelected>>", update_input_fields)

# Accuracy Label
accuracy_label = tk.Label(root, text="", font=("Arial", 11, "italic"), bg="#f0f4f7", fg="#555")
accuracy_label.pack(pady=5)

# Input Frame
input_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove", padx=10, pady=10)
input_frame.pack(pady=10, fill="both", expand=True)

input_entries = []

# Predict Button
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#f0f4f7", fg="green")
result_label.pack(pady=5)

# Train models on startup
train_models()

root.mainloop()
