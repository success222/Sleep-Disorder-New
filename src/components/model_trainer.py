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
                "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
                "SVC": SVC(class_weight='balanced', kernel='rbf', random_state=42),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
                "Extra Tree": ExtraTreeClassifier(class_weight='balanced', random_state=42)
            }
            
            
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                                 X_test=X_test, y_test=y_test, 
                                                 models=models)
            
            ## Get the best model from the dict
            best_model_score = max(sorted(model_report.values()))
            
            ## Get the name of best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with accuracy score: {best_model_score}")
            
            
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
            
        except Exception as e:
            raise CustomException(e, sys)