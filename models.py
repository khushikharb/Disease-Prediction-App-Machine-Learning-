from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, 
                           confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

def evaluate_model(model, X_test, y_test, model_name, le):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'confusion_matrix_path': f'confusion_matrix_{model_name}.png'
    }

def train_and_evaluate(X_train, X_test, y_train, y_test, le, algorithm):
    results = {}
    
    if algorithm == "KMeans":
        # KMeans is unsupervised - we'll use it differently
        n_clusters = len(le.classes_)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X_train)
        
        # For evaluation, we'll assign clusters to most common class
        y_pred = model.predict(X_test)
        
        # Create a mapping from cluster to class
        cluster_to_class = {}
        for cluster in range(n_clusters):
            mask = (y_pred == cluster)
            if sum(mask) > 0:
                cluster_to_class[cluster] = np.bincount(y_test[mask]).argmax()
        
        # Convert cluster predictions to class predictions
        y_pred_mapped = np.array([cluster_to_class[c] for c in y_pred])
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_mapped)
        f1 = f1_score(y_test, y_pred_mapped, average='weighted')
        cm = confusion_matrix(y_test, y_pred_mapped)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=le.classes_, 
                    yticklabels=le.classes_)
        plt.title('Confusion Matrix - KMeans')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_KMeans.png')
        plt.close()
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'confusion_matrix_path': 'confusion_matrix_KMeans.png',
            'model': model
        }
        
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, "Decision Tree", le)
        results['model'] = model
        
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, "Random Forest", le)
        results['model'] = model
        
    elif algorithm == "SVM":
        model = SVC(probability=True, random_state=42)
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, "SVM", le)
        results['model'] = model
        
    else:
        raise ValueError("Invalid algorithm selected")
    
    return results