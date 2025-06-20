import mlflow
import joblib
import mlflow.sklearn
from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
from sklearn.emsemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub

# Initialize DAGsHub tracking
dagshub.init(repo_owner='pkolekar940', repo_name='mlflow-dagshub', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/pkolekar940/mlflow-dagshub.mlflow")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model hyperparameter
max_depth = 15

# Start MLflow run
with mlflow.start_run():

    # Train model
    # dt = DecisionTreeClassifier(max_depth=max_depth)
    # dt.fit(X_train, y_train)
    rf = RandomForestClassifier(max_depth=max_depth,  random_state=42)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Save model using joblib instead of triggering registry
    joblib.dump(rf, "decision_tree_model.pkl")
    mlflow.log_artifact("decision_tree_model.pkl")

    # Log current script if __file__ is available
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        print("⚠️ __file__ is not defined; skipping script logging.")

    # Tags
    mlflow.set_tag('author', 'nitish')
    mlflow.set_tag('model', 'decision tree')

    print('accuracy', accuracy)
