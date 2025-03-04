import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve, auc

# Setup MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("sentiment-analysis")

# Define parameters
params = {
    'vectorizer__max_features': 5000,
    'vectorizer__stop_words': 'english',
    'vectorizer__ngram_range': (1, 2),
    'classifier__solver': 'saga',
    'classifier__max_iter': 1000
}

with mlflow.start_run():
    mlflow.set_tag("developer", "Sourav")
    
    # Step 1: Load the dataset
    try:
        df = pd.read_csv('data/IMDB-Dataset.csv')
        mlflow.log_artifact("data/IMDB-Dataset.csv")
    except FileNotFoundError:
        print("Dataset file not found!")
        raise

    # Sample the data to reduce size for quick training
    df = df.sample(n=10000, random_state=42)
    mlflow.log_param("sample_size", 10000)
    mlflow.log_param("random_state", 42)

    # Step 2: Prepare Data
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Step 3: Split into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mlflow.log_param("test_size", 0.2)

    # Step 4: Create and Train Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])

    # Set parameters
    pipeline.set_params(**params)
    
    # Log all parameters
    mlflow.log_params(params)

    # Train model
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate Model
    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "auc_roc": auc(fpr, tpr),
        "auc_pr": auc(recall, precision)
    })

    # Generate and log classification report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Save the model using MLflow
    input_example = X_test.iloc[:5].to_frame()  # Convert to DataFrame and use the first 5 rows as an example
    mlflow.sklearn.log_model(pipeline, "sentiment_model", input_example=input_example)

    # Print results
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Cross-validation Score: {cv_scores.mean():.2f} (+/- {cv_scores.std()*2:.2f})")
    print(f"Classification Report:\n{report}")

