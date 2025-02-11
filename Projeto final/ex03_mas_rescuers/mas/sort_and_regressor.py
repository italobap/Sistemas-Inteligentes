import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Load datasets
df_train = pd.read_csv(r"C:\Users\italo\OneDrive\Documentos\GitHub\Sistemas-Inteligentes\Projeto final\ex03_mas_rescuers\mas\env_vital_signals_4000v.txt", header=None)
df_test = pd.read_csv(r"C:\Users\italo\OneDrive\Documentos\GitHub\Sistemas-Inteligentes\Projeto final\ex03_mas_rescuers\mas\env_vital_signals_800v.txt", header=None)

# Define column names
columns = ["Id", "pSist", "pDiast", "qPA", "pulse", "respiratory_rate", "severity", "severity_class"]
df_train.columns = columns
df_test.columns = columns

# Drop unnecessary columns
df_train = df_train.drop(columns=["Id", "pSist", "pDiast"])
df_test = df_test.drop(columns=["Id", "pSist", "pDiast"])

# Separate features and targets
X_train = df_train.drop(columns=["severity", "severity_class"])
y_class_train = df_train["severity_class"]
y_reg_train = df_train["severity"]

X_test = df_test.drop(columns=["severity", "severity_class"])
y_class_test = df_test["severity_class"]
y_reg_test = df_test["severity"]

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter grids
cart_class_params = {"max_depth": [3, 5, 10], "min_samples_split": [5, 10, 20], "criterion": ["gini", "entropy"]}
perceptron_class_params = {"max_iter": [500, 1000, 2000], "eta0": [0.001, 0.01, 0.1], "penalty": [None, "l2"]}
cart_reg_params = {"max_depth": [3, 5, 10], "min_samples_split": [5, 10, 20], "criterion": ["squared_error"]}
perceptron_reg_params = {"max_iter": [500, 1000, 2000], "eta0": [0.001, 0.01, 0.1], "penalty": [None, "l2"]}

# Train models using Grid Search
cart_classifier = GridSearchCV(DecisionTreeClassifier(), cart_class_params, cv=5, scoring="accuracy", n_jobs=-1, return_train_score=True)
cart_classifier.fit(X_train_scaled, y_class_train)

perceptron_classifier = GridSearchCV(Perceptron(), perceptron_class_params, cv=5, scoring="accuracy", n_jobs=-1, return_train_score=True)
perceptron_classifier.fit(X_train_scaled, y_class_train)

cart_regressor = GridSearchCV(DecisionTreeRegressor(), cart_reg_params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True)
cart_regressor.fit(X_train_scaled, y_reg_train)

perceptron_regressor = GridSearchCV(Perceptron(), perceptron_reg_params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True)
perceptron_regressor.fit(X_train_scaled, y_reg_train > 50)  # Binary regression for Perceptron

# Evaluate best classifiers
y_class_pred_cart = cart_classifier.best_estimator_.predict(X_test_scaled)
y_class_pred_perceptron = perceptron_classifier.best_estimator_.predict(X_test_scaled)

# Evaluate best regressors
y_reg_pred_cart = cart_regressor.best_estimator_.predict(X_test_scaled)
y_reg_pred_perceptron = perceptron_regressor.best_estimator_.predict(X_test_scaled)

# Compute classification metrics
metrics_cart = precision_recall_fscore_support(y_class_test, y_class_pred_cart, average='weighted')
metrics_perceptron = precision_recall_fscore_support(y_class_test, y_class_pred_perceptron, average='weighted')

# Extract top 3 configurations for each model
cart_class_top3 = pd.DataFrame(cart_classifier.cv_results_).nlargest(3, "mean_test_score")
perceptron_class_top3 = pd.DataFrame(perceptron_classifier.cv_results_).nlargest(3, "mean_test_score")
cart_reg_top3 = pd.DataFrame(cart_regressor.cv_results_).nsmallest(3, "mean_test_score")
perceptron_reg_top3 = pd.DataFrame(perceptron_regressor.cv_results_).nsmallest(3, "mean_test_score")

# Display results
pd.set_option("display.max_colwidth", None)
print("Top 3 configurations for CART Classifier:")
print(cart_class_top3[["params", "mean_test_score"]])
print("\nTop 3 configurations for Perceptron Classifier:")
print(perceptron_class_top3[["params", "mean_test_score"]])
print("\nTop 3 configurations for CART Regressor:")
print(cart_reg_top3[["params", "mean_test_score"]])
print("\nTop 3 configurations for Perceptron Regressor:")
print(perceptron_reg_top3[["params", "mean_test_score"]])

# Compute and print classification results
results_df = pd.DataFrame([
    {"Model": "CART Classifier", "Accuracy": accuracy_score(y_class_test, y_class_pred_cart), "Precision": metrics_cart[0], "Recall": metrics_cart[1], "F1-Score": metrics_cart[2]},
    {"Model": "Perceptron Classifier", "Accuracy": accuracy_score(y_class_test, y_class_pred_perceptron), "Precision": metrics_perceptron[0], "Recall": metrics_perceptron[1], "F1-Score": metrics_perceptron[2]}
])
print(results_df)

# Compute and print regression results
regression_results = pd.DataFrame({"Model": ["CART Regressor", "Perceptron Regressor"], "MSE": [mean_squared_error(y_reg_test, y_reg_pred_cart), mean_squared_error(y_reg_test > 50, y_reg_pred_perceptron)]})
print("\nRegression Results:")
print(regression_results)

# Display confusion matrices
print("CART Classifier Confusion Matrix and Report:")
ConfusionMatrixDisplay.from_predictions(y_class_test, y_class_pred_cart)
print(classification_report(y_class_test, y_class_pred_cart))

print("\nPerceptron Classifier Confusion Matrix and Report:")
ConfusionMatrixDisplay.from_predictions(y_class_test, y_class_pred_perceptron)
print(classification_report(y_class_test, y_class_pred_perceptron))

# Plot regression ConfusionMatrix results
# plt.tight_layout()
# plt.show()
