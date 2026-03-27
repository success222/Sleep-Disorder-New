import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():
            param_dist = params[model_name]

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=10,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=42
            )

            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            y_test_pred = best_model.predict(X_test)

            test_f1_score = f1_score(y_test, y_test_pred, average="weighted")

            report[model_name] = test_f1_score
            best_models[model_name] = best_model

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)