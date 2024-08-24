from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow
import mlflow.sklearn
import os

# Load the dataset
df = pd.read_csv('E:/mlfow-demo/advance-mlfow/diabetes.csv')
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Splitting the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

# Set the experiment name in MLflow
mlflow.set_experiment('diabetes-rf-hyp')

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)

# Start the MLflow run
with mlflow.start_run():
    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Log all params for each combination
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy', grid_search.cv_results_['mean_test_score'][i])

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log best parameters
    mlflow.log_params(best_params)

    # Log the best score
    mlflow.log_metric('accuracy', best_score)

    # Log training and validation data
    train_df = X_train.copy()
    train_df['Outcome'] = y_train
    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    train_csv_path = "train_data.csv"
    test_csv_path = "test_data.csv"
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    mlflow.log_artifact(train_csv_path, 'train data')
    mlflow.log_artifact(test_csv_path, 'validation data')

    # Log source code file, if running as a script
    if '__file__' in globals():
        mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")

    # Set additional tags
    mlflow.set_tag("Author", "Shahbaz Zulfiqar")

    # Print the best parameters and the best cross-validation score
    print("Best Parameters:", best_params)
    print("Best Cross-Validation Score:", best_score)

# Cleanup temporary files
# os.remove(train_csv_path)
# os.remove(test_csv_path)
