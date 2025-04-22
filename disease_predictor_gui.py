import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Disease Prediction System")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.raw_df = None
        self.model = None
        self.le = LabelEncoder()
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.symptoms_list = []
        self.symptom_vars = []
        
        # Create GUI
        self.create_widgets()
    
    def create_widgets(self):
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Data Loading Tab
        self.load_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.load_tab, text="1. Data Loading")
        self.create_data_loading_tab()
        
        # Modeling Tab
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="2. Model Training")
        self.create_model_tab()
        
        # Prediction Tab
        self.pred_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_tab, text="3. Disease Prediction")
        self.create_prediction_tab()
        
        # Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="4. Results & Analysis")
        self.create_results_tab()
    
    def create_data_loading_tab(self):
        # File upload section
        upload_frame = ttk.LabelFrame(self.load_tab, text="Data Upload", padding=10)
        upload_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(upload_frame, text="Browse CSV File", command=self.load_dataset).pack(side=tk.LEFT)
        self.file_label = ttk.Label(upload_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
    
    def create_model_tab(self):
        # Model training section
        train_frame = ttk.LabelFrame(self.model_tab, text="Model Training", padding=10)
        train_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Algorithm selection
        ttk.Label(train_frame, text="Select Algorithm:").grid(row=0, column=0, sticky=tk.W)
        self.algorithm = tk.StringVar(value="Random Forest")
        algorithms = ["Random Forest", "Decision Tree", "SVM"]
        for i, algo in enumerate(algorithms):
            ttk.Radiobutton(train_frame, text=algo, variable=self.algorithm, 
                          value=algo).grid(row=0, column=i+1, sticky=tk.W)
        
        # Train button
        ttk.Button(train_frame, text="Train Model", command=self.train_model
                  ).grid(row=1, column=0, columnspan=4, pady=10)
        
        # Model info
        self.model_info_text = tk.Text(self.model_tab, height=10, state=tk.DISABLED)
        self.model_info_text.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
    
    def create_prediction_tab(self):
        # Symptom selection
        symptom_frame = ttk.LabelFrame(self.pred_tab, text="Select Symptoms", padding=10)
        symptom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Will be populated after data loading
        self.symptom_selection_frame = ttk.Frame(symptom_frame)
        self.symptom_selection_frame.pack()
        
        # Prediction button
        ttk.Button(symptom_frame, text="Predict Disease", command=self.predict
                  ).pack(pady=10)
        
        # Prediction results
        self.prediction_result = ttk.Label(self.pred_tab, text="", 
                                         font=('Helvetica', 12), wraplength=600)
        self.prediction_result.pack(pady=10)
        
        # Probability visualization
        self.probability_frame = ttk.Frame(self.pred_tab)
        self.probability_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def create_results_tab(self):
        # Results display
        results_frame = ttk.Frame(self.results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Text results
        self.results_text = tk.Text(results_frame, height=15, state=tk.DISABLED)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Visualization frame
        self.viz_frame = ttk.Frame(results_frame)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Confusion matrix button
        ttk.Button(self.results_tab, text="Show Confusion Matrix", 
                 command=self.show_confusion_matrix).pack(pady=5)
    
    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
            
        try:
            self.raw_df = pd.read_csv(file_path)
            self.file_label.config(text=file_path.split('/')[-1])
            
            # Get symptom columns (excluding target column)
            self.symptoms_list = self.raw_df.columns[:-1].tolist()
            self.update_symptom_selection()
            
            # Update raw data preview (skipping detailed preview for now)
            messagebox.showinfo("Success", "Dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def update_symptom_selection(self):
        # Clear old symptom checkboxes
        for widget in self.symptom_selection_frame.winfo_children():
            widget.destroy()
        
        # Create new symptom checkboxes based on dataset columns
        self.symptom_vars = []
        for symptom in self.symptoms_list:
            var = tk.IntVar()
            self.symptom_vars.append(var)
            ttk.Checkbutton(self.symptom_selection_frame, text=symptom, variable=var).pack(anchor=tk.W)
    
    def train_model(self):
        if self.raw_df is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
            
        try:
            X = self.raw_df.iloc[:, :-1].values
            y = self.le.fit_transform(self.raw_df.iloc[:, -1])
            
            # Split data
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Train selected model
            algorithm = self.algorithm.get()
            if algorithm == "Random Forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == "Decision Tree":
                self.model = DecisionTreeClassifier(random_state=42)
            elif algorithm == "SVM":
                self.model = SVC(probability=True, random_state=42)
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            self.y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            
            # Display model info
            self.model_info_text.config(state=tk.NORMAL)
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(tk.END, f"Model: {algorithm}\n")
            self.model_info_text.insert(tk.END, f"Accuracy: {accuracy:.2%}\n\n")
            self.model_info_text.insert(tk.END, "Now you can make predictions in the Prediction tab")
            self.model_info_text.config(state=tk.DISABLED)
            
            messagebox.showinfo("Success", f"{algorithm} model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def predict(self):
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please train a model first")
            return
        
        try:
            # Create symptom vector
            symptom_vector = np.zeros(len(self.symptoms_list))
            for i, var in enumerate(self.symptom_vars):
                symptom_vector[i] = var.get()
        
            # Make prediction
            prediction = self.model.predict([symptom_vector])
            disease = self.le.inverse_transform(prediction)[0]
        
            # Get probabilities
            probabilities = self.model.predict_proba([symptom_vector])[0]
            n_classes = len(self.le.classes_)
            top_n = min(3, n_classes)  # Don't try to get more classes than exist
            top_indices = np.argsort(probabilities)[-top_n:][::-1]
            top_diseases = self.le.inverse_transform(top_indices)
            top_probs = probabilities[top_indices]
        
            # Display prediction
            prediction_text = (f"Most Likely Disease: {disease}\n\n"
                               f"Top {top_n} Possible Diseases:\n")
            for i in range(top_n):
                prediction_text += f"{i+1}. {top_diseases[i]} ({top_probs[i]:.1%})\n"
        
            self.prediction_result.config(text=prediction_text, foreground="green")
        
            # Plot probabilities
            self.plot_probabilities(top_diseases, top_probs)
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
    def plot_probabilities(self, diseases, probabilities):
        # Clear previous plot
        for widget in self.probability_frame.winfo_children():
            widget.destroy()
        
        # Create new plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(diseases, probabilities, color="skyblue")
        ax.set_xlabel("Probability")
        ax.set_title("Top Disease Probabilities")
        
        # Show plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.probability_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_confusion_matrix(self):
        if self.y_pred is None or self.y_test is None:
            messagebox.showerror("Error", "Please train a model and make predictions first")
            return

    # Clear previous confusion matrix plot
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

    # Compute confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

    # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=self.le.classes_, yticklabels=self.le.classes_)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

    # Show in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()
