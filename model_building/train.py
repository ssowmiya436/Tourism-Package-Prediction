# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction-experiment")

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Replace with your Hugging Face username
hf_username = "ssowmiya"

# Load train and test data from Hugging Face
Xtrain_path = f"hf://datasets/{hf_username}/tourism-package-prediction/Xtrain.csv"
Xtest_path = f"hf://datasets/{hf_username}/tourism-package-prediction/Xtest.csv"
ytrain_path = f"hf://datasets/{hf_username}/tourism-package-prediction/ytrain.csv"
ytest_path = f"hf://datasets/{hf_username}/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print("Training and test data loaded successfully from Hugging Face.")
print(f"Training set size: {Xtrain.shape[0]}, Test set size: {Xtest.shape[0]}")

# Define numerical and categorical features
numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
                    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']

categorical_features = ['TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus',
                        'ProductPitched', 'Designation']

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'xgbclassifier__colsample_bytree': [0.5, 0.7, 0.9],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)
    print(f"Best parameters: {grid_search.best_params_}")

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Generate classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    print("\n--- Training Set Performance ---")
    print(classification_report(ytrain, y_pred_train))

    print("\n--- Test Set Performance ---")
    print(classification_report(ytest, y_pred_test))

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1_score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1_score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face Model Hub
    model_repo_id = f"{hf_username}/tourism-package-model"
    repo_type = "model"

    # Check if the model repository exists
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Model repository '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{model_repo_id}' not found. Creating new repository...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
        print(f"Model repository '{model_repo_id}' created.")

    # Upload the model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=model_repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {model_repo_id}")

print("Model training and registration completed successfully!")
