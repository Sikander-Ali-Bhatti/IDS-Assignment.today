#Name sikander Ali 
#ROll No:23BSAI-37
#Assignment no:02
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from itertools import product

# IMAGE DATASET: MNIST

print("Loading MNIST dataset...")

mnist = datasets.fetch_openml('mnist_784', version=1)
X_img = mnist.data
y_img = mnist.target.astype(int)

# Normalize pixel values
X_img = X_img / 255.0


X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42
)

# TEXT/FEATURE DATASET: UCI Digits

print("Loading UCI Digits dataset...")

digits = datasets.load_digits()
X_txt = digits.data
y_txt = digits.target


scaler = StandardScaler()
X_txt = scaler.fit_transform(X_txt)

X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    X_txt, y_txt, test_size=0.2, random_state=42
)


kernels = ['linear', 'rbf', 'poly', 'sigmoid']
C_values = [0.1, 1]
gamma_values = ['scale', 0.01]

results = []

def run_svm(X_train, X_test, y_train, y_test, dataset_name):
    print(f"\nRunning SVM on {dataset_name} dataset\n")
    
    for kernel, C, gamma in product(kernels, C_values, gamma_values):
        
        if kernel == 'linear':
            model = SVC(kernel=kernel, C=C)
        else:
            model = SVC(kernel=kernel, C=C, gamma=gamma)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Kernel: {kernel}, C: {C}, Gamma: {gamma} → Accuracy: {acc:.4f}")
        
        results.append({
            "Dataset": dataset_name,
            "Kernel": kernel,
            "C": C,
            "Gamma": gamma,
            "Accuracy": acc
        })

run_svm(X_train_img, X_test_img, y_train_img, y_test_img, "MNIST")
run_svm(X_train_txt, X_test_txt, y_train_txt, y_test_txt, "UCI Digits")

results_df = pd.DataFrame(results)

print("\nSummary of Results:")
print(results_df)