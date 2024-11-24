import logging
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
) 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Evaluation Strategy
# ----------------------------------------------------
# This class defines a common interface for different model evaluation strategies.
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model:ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series
    ) -> Dict[str, float]:
        """
        Abstract method to evaluate a model's performance.

        Parameters:
            model (ClassifierMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        pass

# Concrete Strategy for Classification Model Evaluation
# -----------------------------------------------------
# This class implements the ModelEvaluationStrategy for classification models.
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates a classification model using various metrics.

        Parameters:
            model (ClassifierMixin): The trained classification model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix.
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        F1_score = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        Confusion_matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": F1_score,
            "ROC AUC": roc_auc,
            "True Negatives": Confusion_matrix[0][0],
            "False Positives": Confusion_matrix[0][1],
            "False Negatives": Confusion_matrix[1][0],
            "True Positives": Confusion_matrix[1][1],

        }
        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics
    
# Context Class for Model Evaluation
# -----------------------------------
# This class uses a specific evaluation strategy to assess the performance of a model.
class ModelEvaluator:
    def __init__(self, strategy:ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
            strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
            strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
            model (ClassifierMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the Classification Model Evaluation strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)
    

# Example usage
if __name__ == "__main__":
    # # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_classification_model
    # X_test = test_data_features
    # y_test = test_data_target

    # # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(ClassificationModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics) 
    pass