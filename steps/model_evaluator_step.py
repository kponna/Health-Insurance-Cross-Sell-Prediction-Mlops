import mlflow
import logging
from typing import Dict 
import pandas as pd
from sklearn.pipeline import Pipeline 
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy

from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

logger = get_logger(__name__)

experiment_tracker=Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_evaluation_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Step to evaluate a classification model.

    Parameters:
        trained_model (ClassifierMixin): The trained classification model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """ 
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")
    
    try:
        logging.info("Applying the same preprocessing to the test data.")
        preprocessor = trained_model.named_steps.get('preprocessor', None)
        logging.info(f"Evaluating model with parameters: {trained_model['model'].get_params()}")
        if preprocessor:
            # Log the columns being scaled if a scaler exists in the 'num' transformer
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':  # Check if the transformer is for numerical columns
                    logging.info(f"Numerical columns {cols} are being scaled using the scaler in the pipeline.")
   
            # Apply the preprocessing and scaling to the test data
            X_test_preprocessed  = trained_model.named_steps['preprocessor'].transform(X_test)
            # print("First 5 rows of X_test_processed:")
            # print(X_test_preprocessed[:5]) 
            
        else:
            logging.warning("No 'preprocessor' step found in the pipeline. Skipping preprocessing.")
            X_test_preprocessed  = X_test   

        evaluator = ModelEvaluator(ClassificationModelEvaluationStrategy())
        metrics = evaluator.evaluate(trained_model.named_steps["model"], X_test_preprocessed, y_test)

        # log metrics to mlflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        return metrics
        
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}")
        raise e     