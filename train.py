import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,confusion_matrix
import  matplotlib.pyplot as plt
import seaborn as sns 


iris=load_iris()
X=iris.data
y=iris.target  

# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

max_depth=20
n_estimator=50

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimator,random_state=42)
    rf.fit(X_train,y_train)
    # make prediction
    y_pred=rf.predict(X_test)
    accuracy_value=accuracy_score(y_test,y_pred)
    
    