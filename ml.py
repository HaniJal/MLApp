import os
import pickle
import csv
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = fetch_california_housing()
REG_MODEL_FILE = "regression_california_linear.pkl"
PHISHING_CSV = "dataset.csv" 
CLF_MODEL_FILES = {
    1: "classification_phishing_knn.pkl",
    2: "classification_phishing_svm.pkl",
    3: "classification_phishing_dt.pkl",
}


def load_phishing_csv(filename)-> Tuple[List[List[float]], List[int], List[str]]:
    x: List[List[float]] = []
    y: List[int] = []
    feature_names: List[str] = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        
        feature_names = header[1:-1]
        target_name = header[-1].strip().lower()
        if target_name != "result":  
            raise ValueError("Target name is wrong. Check csv file.")
        for row in reader:
            if not row:
                continue
            values = [float(v) for v in row]
            x.append(values[1:-1])
            y.append(int(values[-1]))
        return x,y, feature_names

def split_train_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.20, random_state=42)
    print("\nTrain/test split:")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    return x_train, x_test, y_train, y_test

def build_regression_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    test_mae = mean_absolute_error(y_test, y_predict)
    test_mse = mean_squared_error(y_test, y_predict)
    test_r2 = r2_score(y_test, y_predict)
    print("\nFinal Test Set Evaluation:")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R^2: {test_r2:.4f}")

    return test_mae, test_mse, test_r2

def train_regression():
    print("\nLoading California Housing dataset...")
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_regression_pipeline()
    train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

    # Save the ENTIRE pipeline
    with open(REG_MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to: {REG_MODEL_FILE}\n")


def predict_regression():
    if not os.path.exists(REG_MODEL_FILE):
        print("\nModel not found. Please train the model first.\n")
        return

    with open(REG_MODEL_FILE, "rb") as f:
        pipeline = pickle.load(f)

    feature_names = data.feature_names

    print(f"\nLoading model: {REG_MODEL_FILE}")
    row = []
    for name in feature_names:
        val = ask_float(f"Enter value for '{name}': ")
        row.append(val)

    pred = pipeline.predict([row])[0]
    print(f"\nEstimated house value: {pred:.4f}\n")


def build_classification_pipeline(clf, k=None, C=None, max_depth=None):
    if clf == 1:  # KNN
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=k))
        ])
    elif clf == 2:  # SVM
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=C))
        ])
    else:  # Decision Tree
        return Pipeline([
            ("model", DecisionTreeClassifier(max_depth=max_depth, random_state=42))
        ])
    
def evaluate_classifier(model, x_test, y_test):
    y_predict = model.predict(x_test)

    acc = accuracy_score(y_test, y_predict)
    prec = precision_score(y_test, y_predict, pos_label=1, zero_division=0)
    rec  = recall_score(y_test, y_predict, pos_label=1, zero_division=0)
    f1   = f1_score(y_test, y_predict, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_predict)

    print("\nFinal Test Set Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return acc, prec, rec, f1, cm

def train_classification(clf):
    # clf is 1, 2, or 3 (int)

    if clf == 1:  # KNN
        k = ask_int("Enter k (number of neighbors): ")
        C = None
        max_depth = None

    elif clf == 2:  # SVM
        k = None
        C = ask_float("Enter C (SVM regularization): ")
        max_depth = None

    else:  # clf == 3  Decision Tree
        k = None
        C = None
        max_depth = ask_int("Enter max_depth (Decision Tree): ")

    print("\nLoading phishing dataset from CSV ...")
    x, y, feature_names = load_phishing_csv(PHISHING_CSV)

    x_train, x_test, y_train, y_test = split_train_test(x, y)

    pipeline = build_classification_pipeline(clf, k=k, C=C, max_depth=max_depth)

    pipeline.fit(x_train, y_train)
    evaluate_classifier(pipeline, x_test, y_test)

    model_file = CLF_MODEL_FILES[clf]
    with open(model_file, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to: {model_file}\n")

def predict_classification(clf):
    model_file = CLF_MODEL_FILES[clf]

    if not os.path.exists(model_file):
        print("\nModel not found. Please train the model first.\n")
        return

    with open(model_file, "rb") as f:
        pipeline = pickle.load(f)

    _, _, feature_names = load_phishing_csv(PHISHING_CSV)

    print(f"\nLoading model: {model_file}")
    row = []
    for name in feature_names:
        val = ask_int(f"Enter feature value for '{name}' (-1/0/1): ")
        row.append(val)

    pred = pipeline.predict([row])[0]

    if pred == 1:
        label = "phishing"
    elif pred == -1:
        label = "legitimate"
    else:
        label = f"unknown (raw={pred})"
    
    print(f"\nPrediction:\nClass: {label} (raw={pred})\n")

def choose_task():
    while True:
        print("\n Please enter your choice:")
        print("1. Regression (estimate a number)")
        print("2. Classification (phishing detection)")
        choice = input("> ").strip()
        if choice in {"1","2"}:
            return choice
        print("Invalid choice. Please enter 1 or 2.")
def choose_action():
    while True:
        print("\n Choose option :")
        print("1. Train model")
        print("2. Use model for prediction")
        action = input("> ").strip()
        if action in {"1","2"}:
            return action
        print("Invalid choice. Please enter 1 or 2.")

def choose_classifier():
    while True:
        print("\nChoose classifier:")
        print("1) KNN")
        print("2) SVM")
        print("3) Decision Tree")
        choice = input("> ").strip()

        if choice in {"1", "2", "3"}:
            return int(choice)

        print("Invalid choice. Please enter 1, 2, or 3.")

def ask_int(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return int (value)
        except ValueError:
            print("please enter valid integer!")
            
def ask_float(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("Please enter a valid number.")

def main():

    task = choose_task()
    action = choose_action()
    if task == "1":
        if action == "1":
            train_regression()
        else:
            predict_regression()

    else:  # Classification
        clf = choose_classifier()  # 1,2,3 (only needed for classification)
        if action == "1":
            train_classification(clf)
        else:
            predict_classification(clf) 

if __name__=="__main__":
    main()
