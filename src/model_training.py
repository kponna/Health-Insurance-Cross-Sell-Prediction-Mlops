import os
import logging
import joblib
import pandas as pd
from typing import Any
from abc import ABC, abstractmethod
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Training Strategy
# ------------------------------------------------
# This class defines a common interface for model training strategies.
class ModelTrainingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool) -> Pipeline:
        """
        Abstract method to build and train a model. 
        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Whether to perform hyperparameter tuning. 
        Returns:
            Pipeline: A trained model pipeline.
        """
        pass


# Concrete Strategy for Generalized Model Training
# ------------------------------------------------
# This class implements the ModelTrainingStrategy interface and trains 
class GeneralizedModelTrainingStrategy(ModelTrainingStrategy):
    def __init__(self, model, param_grid=None):
        """
        Initializes the strategy with a specific model and parameter grid for hyperparameter tuning.

        Parameters:
            model: The machine learning model to use for training.
            param_grid (dict, optional): The parameter grid for hyperparameter tuning.
        """
        self.model = model
        self.param_grid = param_grid

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool) -> Pipeline:
        """
        Builds and trains the model, with an option for hyperparameter tuning.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels.
            fine_tuning (bool): Whether to perform hyperparameter tuning.

        Returns:
            Pipeline: The trained model pipeline.
        """
        logging.info(f"Training {self.model.__class__.__name__} model.")

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.") 
        try: 
            model_name = self.model.__class__.__name__
            # Log model parameters and type before training
            mlflow.log_param("model_type", model_name) 
            if fine_tuning and self.param_grid is not None:
                logging.info(f"Performing hyperparameter tuning with parameters on sampled dataset: {self.param_grid}.")
                X_sampled, _, y_sampled, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42)
                best_model_pipeline = self.perform_hyperparameter_tuning(X_sampled, y_sampled) 
                logging.info("Finished hyperparameter tuning.")
            else:
                best_params = {'penalty': 'elasticnet', 'l1_ratio': 0.5, 'C': 1} # Replace these values with best tuned parameters of the model after experimenting
                self.model.set_params(**best_params)
                logging.info("Fitting model with provided parameters.")
                best_model_pipeline = self.model.fit(X_train, y_train)
                logging.info(f"Training with parameters: {self.model.get_params()}") 

                self.save_model(best_model_pipeline)
                    
            return best_model_pipeline    
        
        except Exception as e:
            logging.error(f"An error occurred in model training: {e}")
            raise e
  
    def perform_hyperparameter_tuning(self, X_sampled: pd.DataFrame, y_sampled: pd.Series) -> Pipeline:
        """
        Performs hyperparameter tuning using RandomizedSearchCV.

        Parameters:
            X_sampled (pd.DataFrame): The sampled training data features.
            y_sampled (pd.Series): The sampled training data labels/target.

        Returns:
            Pipeline: The best trained model after hyperparameter tuning.
        """ 
        randomized_search = RandomizedSearchCV(
            self.model,
            param_distributions=self.param_grid,
            n_iter=10,
            cv=StratifiedKFold(n_splits=3),
            n_jobs=-1,
            # verbose=2,
            scoring='recall'
        )
        randomized_search.fit(X_sampled, y_sampled) 
        logging.info(f"Best parameters: {randomized_search.best_params_}")
        logging.info(f"Best recall score: {randomized_search.best_score_}")
        return randomized_search.best_estimator_

    def save_model(self, model_pipeline: Pipeline) -> None:
        """
        Saves the trained model pipeline to disk.

        Parameters:
            model_pipeline (Pipeline): The trained model pipeline.
        """
        os.makedirs('data/artifacts/models', exist_ok=True)
        model_name = self.model.__class__.__name__.lower()  
        joblib.dump(model_pipeline, f'data/artifacts/models/{model_name}_pipeline.pkl')
        logging.info(f"{model_name.capitalize()} pipeline saved.")

# Context Class for Model Training class to use different strategies
# ------------------------------------------------
# This class is responsible for using the defined strategy to build and train the model.
class ModelTrainer:
    def __init__(self, strategy: ModelTrainingStrategy)-> None:
        """
        Initializes the ModelTrainer with a specific model training strategy.

        Parameters:
            strategy (ModelTrainingStrategy): The strategy to be used for building and training the model.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelTrainingStrategy)-> None:
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
            strategy (ModelTrainingStrategy): The new strategy to be used for model training.
        """
        self._strategy = strategy

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool) -> Any:
        """
        Executes the model building and training using the current strategy.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
            fine_tuning (bool): Whether to perform hyperparameter tuning.

        Returns:
            Any: The trained model pipeline.
        """
        return self._strategy.build_and_train_model(X_train, y_train, fine_tuning)

# Example usage:
if __name__ == "__main__":
    # # Define parameter grids for different models if fine-tuning is required
    # logistic_param_grid = {'C': [0.1, 0.5, 1, 5], 'penalty': ['l1', 'l2', 'elasticnet']}
    # random_forest_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    # decision_tree_param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    # knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']} 
    # gaussian_nb_param_grid = {}  # No hyperparameters for GaussianNB
    # xgboost_param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
    # lightgbm_param_grid = {'num_leaves': [31, 50, 70], 'learning_rate': [0.01, 0.1, 0.2]}

    # # Create instances for different strategies
    # strategies = {
    #     'logistic_regression': GeneralizedModelTrainingStrategy(LogisticRegression(max_iter=2000, random_state=42, solver='saga'), logistic_param_grid),
    #     'random_forest': GeneralizedModelTrainingStrategy(RandomForestClassifier(random_state=42), random_forest_param_grid),
    #     'decision_tree': GeneralizedModelTrainingStrategy(DecisionTreeClassifier(random_state=42), decision_tree_param_grid),
    #     'knn': GeneralizedModelTrainingStrategy(KNeighborsClassifier(), knn_param_grid), 
    #     'gaussian_nb': GeneralizedModelTrainingStrategy(GaussianNB(), gaussian_nb_param_grid),
    #     'xgboost': GeneralizedModelTrainingStrategy(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_param_grid),
    #     'lightgbm': GeneralizedModelTrainingStrategy(LGBMClassifier(), lightgbm_param_grid)
    # }

    # # Assuming you have your training data X_train and y_train prepared
    # X_train, y_train = pd.DataFrame(...), pd.Series(...)
    # for model_name, strategy in strategies.items():
    #     model_trainer = ModelTrainer(strategy)
    #     logging.info(f"Training {model_name} model.")
    #     model_trainer.train(X_train, y_train, fine_tuning=True)  # Set fine_tuning as needed
    pass