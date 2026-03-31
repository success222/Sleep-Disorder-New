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
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def generate_prediction_text(result):
    if result == "Sleep Apnea":
        heading = "You are likely to experience SLEEP APNEA."
        explanation = (
            "Sleep apnea is a sleep disorder in which breathing repeatedly stops "
            "and starts during sleep. It can reduce sleep quality and lead to "
            "daytime tiredness and other health complications."
        )
        advice = (
            "Maintain a healthy weight, avoid alcohol before bedtime, keep a "
            "regular sleep schedule, and seek medical advice for proper evaluation."
        )

    elif result == "Insomnia":
        heading = "You are likely to experience INSOMNIA."
        explanation = (
            "Insomnia is a sleep disorder that makes it difficult to fall asleep, "
            "stay asleep, or get restful sleep. It can affect mood, concentration, "
            "and daily functioning."
        )
        advice = (
            "Keep a consistent bedtime, reduce caffeine intake later in the day, "
            "manage stress, avoid screens before sleep, and speak to a healthcare "
            "professional if symptoms persist."
        )

    else:
        heading = "You are not likely to experience any sleep disorder."
        explanation = (
            "This suggests that your current sleep and lifestyle pattern does not "
            "strongly indicate a sleep disorder at this time."
        )
        advice = (
            "Maintain healthy sleep habits, stay physically active, manage stress, "
            "and continue monitoring your sleep routine to help prevent future sleep problems."
        )

    return heading, explanation, advice