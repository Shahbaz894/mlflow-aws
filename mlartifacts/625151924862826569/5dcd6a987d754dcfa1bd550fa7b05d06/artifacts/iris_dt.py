import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the MLflow tracking URI
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Parameters for the DecisionTreeClassifier
max_depth = 35
n_estimator = 50

# Set the MLflow experiment
mlflow.set_experiment('iris-dt')

# Start an MLflow run
with mlflow.start_run(run_name='mlflow-shahbaz'):
    # Initialize and train the DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = dt.predict(X_test)
    
    # Calculate accuracy
    accuracy_value = accuracy_score(y_test, y_pred)
    
    # Log accuracy and parameters
    mlflow.log_metric('accuracy', accuracy_value)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimator', n_estimator)
    print('Accuracy:', accuracy_value)
   
    # Calculate and plot the confusion matrix
    cn = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cn, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Log the confusion matrix plot as an artifact with MLflow
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.set_tag('author','shahbaz')
    mlflow.set_tag('model','decision tree')
    # sace the code using mlflow
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt, 'decision_tree_model')
