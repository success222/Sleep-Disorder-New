import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered the model trainer method or component")
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
           
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVC": SVC(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Extra Tree": ExtraTreeClassifier(random_state=42)
            }
        
            params = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', 'balanced_subsample']
                },
                "SVC": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced']
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced']
                },
                "Extra Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced']
                }
            }
                
            model_report, best_models = evaluate_models(X_train=X_train, y_train=y_train, 
                                                        X_test=X_test, y_test=y_test, 
                                                        models=models, params=params)
                
            ## Get the best model from the dict
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best model found: {best_model_name} with score: {best_model_score}"
            )
            
            
            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
            
            predicted = best_model.predict(X_test)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')
        
            logging.info(f"Precision Score: {precision}")
            logging.info(f"Recall Score: {recall}")
            logging.info(f"F1 Score: {f1}")
                
            return {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
        except Exception as e:
            raise CustomException(e, sys)