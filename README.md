Practical ML Pipeline (Regression + Classification CLI App)

This project is a command-line Machine Learning application written in Python.
The program behaves like a small ML product: it can train models, evaluate them, save them, and later load them to make predictions based on user input.

The application supports two tasks:

Regression – predicts a continuous numeric value (house price).
Classification – predicts whether a website is phishing or legitimate.

The program saves trained models using pickle so users can reuse trained models without retraining.

How to Run the Program:

Open a terminal in the project folder and run:

python ml.py

The program will show a menu and guide the user through training or prediction.

Menu Flow

When the program starts:

1) Regression (estimate a number)
2) Classification (phishing detection)

Then:

1) Train model
2) Use model for prediction
Regression Mode

Train → trains Linear Regression model on housing data

Predict → asks the user for housing features and estimates house value

Classification Mode

User chooses classifier:

KNN
SVM
Decision Tree

Train → trains model and prints evaluation metrics

Predict → asks user for website features and predicts phishing/legitimate

Datasets Used
1) Regression Dataset – California Housing

Loaded using:
sklearn.datasets.fetch_california_housing()
Target: Median house value
Features include income, house age, rooms, population, latitude, longitude, etc.

2) Classification Dataset – Phishing Website Dataset (Kaggle)

The dataset was loaded using Python’s built-in csv module. The program reads the header row to obtain feature names and converts each value to numeric format before training.

Downloaded from:
https://www.kaggle.com/datasets/akashkr/phishing-website-dataset

Steps:
Logged into Kaggle
Downloaded the ZIP dataset
Extracted the files
Used the CSV file (dataset.csv)
The dataset contains website characteristics such as URL length, SSL state, domain age, etc.

The Result column contains:
1 → phishing
-1 → legitimate

Models Implemented

The regression model (Linear Regression) does not require any user-specified hyperparameters. It is trained using default parameters because Linear Regression automatically computes coefficients directly from the training data.

LinearRegression
Pipeline:
StandardScaler
LinearRegression

Metrics printed:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
R² score


Saved model:
regression_california_linear.pkl

Classification
Supported models:

Model	Hyperparameter asked
KNN	k (number of neighbors)
SVM	C (regularization)
Decision Tree	max_depth

Pipelines:
KNN → StandardScaler + KNeighborsClassifier
SVM → StandardScaler + SVC
Decision Tree → DecisionTreeClassifier

Metrics printed:
Accuracy
Precision
Recall
F1-score
Confusion Matrix

Saved models:
classification_phishing_knn.pkl
classification_phishing_svm.pkl
classification_phishing_dt.pkl
Prediction Behavior

When using prediction mode:
The program loads the saved model 
Prompts the user for each feature value
Builds a single input row
Uses predict() to classify or estimate value
If a model file does not exist, the program prints:
Model not found. Please train the model first.
Discussion


Should the Index Column Be Included in Training?
No.

The first column in the dataset is an index number that only identifies each row.
It does not describe the website itself and has no relationship to phishing detection.
Including it would introduce noise and could mislead the model. Therefore, it is removed before training.

Why Scaling is Used?

Scaling (StandardScaler) is important for KNN and SVM models because they rely on distances between data points. Features in the dataset have different ranges.Without scaling, features with larger numeric values would dominate the distance calculation and negatively affect accuracy.
Decision Trees do not require scaling because they split based on feature thresholds rather than distances.

Why the Pipeline is Saved Using Pickle?

Training a machine learning model can be time-consuming. In real applications, models are trained once and reused many times.
The pickle library allows the entire pipeline (preprocessing + trained model) to be saved to a file. This ensures: 
Users do not need to retrain every time
Predictions use the same preprocessing steps as training
The program behaves like a real software product
Saving the full pipeline is important because the scaler learned parameters (mean and standard deviation) during training, and predictions must use the same transformation.
The entire pipeline (scaler + trained model) is saved, not just the classifier/regressor. This ensures that new prediction data is transformed using the same scaling parameters learned during training.
