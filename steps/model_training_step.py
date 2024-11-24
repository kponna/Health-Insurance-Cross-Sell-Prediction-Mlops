import logging
from typing import Annotated 
import mlflow 
import mlflow.sklearn

import os
import pickle
import pandas as pd  
import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig, step 
from zenml.client import Client 
from src.model_training import GeneralizedModelTrainingStrategy, ModelTrainer
from materializers.customer_materializer import LogisticRegressionPipelineMaterializer, NumpyInt64Materializer 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
 
# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
if experiment_tracker is None:
    raise ValueError("No experiment tracker is configured in the active ZenML stack.")
else:
    logging.info(f"Using experiment tracker: {experiment_tracker.name}")

print(Client().active_stack)

from zenml import Model 
model = Model(
    name="Health_Insurance_Cross_Selling_Predictor",
    version=None,
    license="Apache 2.0",
    description="Predictive model designed to identify potential cross-selling opportunities in health insurance by analyzing customer demographics and behaviors, enabling insurance companies to optimize their marketing strategies and improve customer retention.",
)
 
@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model, output_materializers=[LogisticRegressionPipelineMaterializer,NumpyInt64Materializer])
def model_training_step(
    X_train: pd.DataFrame, y_train: pd.Series, strategy: str, fine_tuning: bool
) -> Annotated[Pipeline, ArtifactConfig(name="trained_model", is_model_artifact=True)]:
    """
    Model building step using ZenML with multiple model options and MLflow tracking.

    Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.
        strategy (str): Model selection method, e.g., 'logistic_regression', 'xgboost', 'svm', 'naive_bayes', 'random_forest'.
        fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

    Returns:
        Trained pipeline instance.
    """   
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
      
    numerical_cols = ['Annual_Premium', 'Vintage']
    categorical_cols = ['Gender', 'Region_Code', 'Age_Encoded', 'Vehicle_Age', 'Policy_Sales_Channel_Encoded', 'Previously_Insured', 'Vehicle_Damage']
    
    logging.info(f"Numerical columns: {numerical_cols}")
    logging.info(f"Categorical columns: {categorical_cols}") 
    
    X_train[numerical_cols] = X_train[numerical_cols].astype('float64')
    
    # Define preprocessing for numerical features 
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    # Define preprocessing for categorical features  
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), 
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    
    logging.info(f"Building model using the strategy: {strategy}")

    # Define parameter grids for different models
    logistic_param_grid = {'C': [0.1, 0.5, 1, 5], 'penalty': ['l1', 'l2', 'elasticnet'], 'l1_ratio': [0.1, 0.5, 0.7]}
    random_forest_param_grid =  {'n_estimators':[50,70,100], "max_depth" : [5,25,50],"min_samples_leaf":[2,10,20]}
    decision_tree_param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    knn_param_grid =   {'n_neighbors':[5,7,9],'weights':['uniform','distance'],'p':[2, 1]} 
    gaussian_nb_param_grid = {}  # No hyperparameters for GaussianNB
    xgboost_param_grid = {'n_estimators': [100, 200, 300],'max_depth': [3, 5, 7],'learning_rate': [0.01, 0.1, 0.2],'subsample': [0.8, 1.0],'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.3],'min_child_weight': [1, 3],'reg_alpha': [0, 0.1],'reg_lambda': [1, 1.5]}
    lightgbm_param_grid = {'n_estimators':[50 ,100], "max_depth" : [25,50], 
                   'min_data_in_leaf':[200,300],'learning_rate':[.001,0.01]}

    # Choose the appropriate strategy 
    if strategy == "logistic_regression":
        if fine_tuning:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(LogisticRegression(max_iter=2000,  solver='saga'), logistic_param_grid)
            )
            logging.info("Implementing Logistic Regression Strategy with Hyperparameter Tuning.")
        else:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(LogisticRegression(max_iter=2000, random_state=42, solver='saga')) 
            )
            logging.info("Implementing Logistic Regression Strategy without Hyperparameter Tuning.")

    elif strategy == "xgboost":
        if fine_tuning:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(XGBClassifier(random_state=5, eval_metric='logloss'), xgboost_param_grid)
            )
            logging.info("Implementing XGBoost Strategy with Hyperparameter Tuning.")
        else:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(XGBClassifier())  
            )
            logging.info("Implementing XGBoost Strategy without Hyperparameter Tuning.")

    elif strategy == "decision_trees":
        if fine_tuning:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(DecisionTreeClassifier(random_state=42), decision_tree_param_grid)
            )
            logging.info("Implementing Decision Trees Strategy with Hyperparameter Tuning.")
        else:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(DecisionTreeClassifier( ))  
            )
            logging.info("Implementing Decision Trees Strategy without Hyperparameter Tuning.")

    elif strategy == "naive_bayes":
        model_builder = ModelTrainer(
            GeneralizedModelTrainingStrategy(GaussianNB(), gaussian_nb_param_grid)
        )
        logging.info("Implementing Naive Bayes Strategy.")

    elif strategy == "random_forest":
        if fine_tuning:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(RandomForestClassifier(), random_forest_param_grid)
            )
            logging.info("Implementing Random Forest Strategy with Hyperparameter Tuning.")
        else:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(RandomForestClassifier())  
            )
            logging.info("Implementing Random Forest Strategy without Hyperparameter Tuning.")

    elif strategy == "lightgbm":
        if fine_tuning:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(LGBMClassifier(random_state=42), lightgbm_param_grid)
            )
            logging.info("Implementing Light Gradient Boosting Machine Strategy with Hyperparameter Tuning.")
        else:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(LGBMClassifier(random_state=42))  # Default parameters
            )
            logging.info("Implementing Light Gradient Boosting Machine Strategy without Hyperparameter Tuning.")

    elif strategy == "knn":
        if fine_tuning:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(KNeighborsClassifier(), knn_param_grid)
            )
            logging.info("Implementing KNN Strategy with Hyperparameter Tuning.")
        else:
            model_builder = ModelTrainer(
                GeneralizedModelTrainingStrategy(KNeighborsClassifier()) 
            )
            logging.info("Implementing KNN Strategy without Hyperparameter Tuning.") 
    else:
        raise ValueError(f"Unknown strategy '{strategy}' selected for model training.")
     
    # Start an MLflow run to log the training process
    if not mlflow.active_run():
        mlflow.start_run() 
    try:
        # Enable autologging to automatically log model parameters, metrics
        mlflow.sklearn.autolog()  
        
        # Create a pipeline 
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_builder.train(X_train, y_train, fine_tuning))])
        # print("Missing values in X_train:", (X_train.isnull().sum()))

        # Train the model
        logging.info("Started model training.")
        pipeline.fit(X_train, y_train) 
        logging.info("Model training has completed.") 
  
        # Save the trained pipeline
        pipeline_path = "/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/models/logisticregression_pipeline.pkl" 
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(pipeline_path, artifact_path="models")

        # Extract the fitted StandardScaler from the pipeline
        scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"] 
        scaler_path = "/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/models/standard_scaler.pkl"
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True) 
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f) 
        logging.info(f"StandardScaler saved in models folder.")
   
    except Exception as e:
        logging.error(f"Model Training failed for strategy: {strategy}, Error: {str(e)}")
        mlflow.log_param("error_details", str(e))
        raise RuntimeError(f"Model training failed for {strategy}") from e
    
    finally:
        # End the mlflow run
        mlflow.end_run()

    return pipeline
