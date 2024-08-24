from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow


# Load the dataset
df = pd.read_csv('E:/mlfow-demo/advance-mlfow/diabetes.csv')

# Splitting the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],  # Corrected typo
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
mlflow.set_experiment('diabtes-rf-hyp')
with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # log params
    mlflow.log_params(best_params)
    # log metrics
    mlflow.log_metric('accuracey',best_score)
    #  log data
    train_df=X_train
    # train_df = pd.DataFrame(X_train, columns=df.feature_names)
    train_df['Outcome'] = y_train
    
    # test_df=X_train
    # test_df['variety']=y_test
    # test_df = pd.DataFrame(X_test, columns=df.feature_names)
    test_df=X_test
    test_df['Outcome'] = y_test
    
    train_df=mlflow.data.from_pandas(train_df)
    test_df=mlflow.data.from_pandas(test_df)
    
    mlflow.log_input(train_df,'train data')
    mlflow.log_input(test_df,'validation data')
    # log source code
    mlflow.log_artifact(__file__)
    
    # log model
    mlflow.sklearn.log_models(grid_search.best_estimator_ , "random forest")
    
    mlflow.set_tag("Author","shahbaz zulfqar")

    print("Best Parameters:", best_params)
    print("Best Cross-Validation Score:", best_score)
