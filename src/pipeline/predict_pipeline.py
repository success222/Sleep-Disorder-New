import sys
import pandas as pd

from src.exception import CustomException
# from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
                 Age: int,
                 Gender: str,
                 Occupation: str,
                 Sleep_Duration: float,
                 Quality_of_Sleep: float,
                 Physical_Activity_Level: float,
                 Stress_Level: float,
                 BMI_Category: str,
                 Systolic_BP: int,
                 Diastolic_BP: int,
                 Heart_Rate: int,
                 Daily_Steps: int):
        self.age = Age
        self.gender = Gender
        self.occupation = Occupation
        self.sleep_duration = Sleep_Duration
        self.quality_of_sleep = Quality_of_Sleep
        self.physical_activity_level = Physical_Activity_Level
        self.stress_level = Stress_Level
        self.bmi_category = BMI_Category
        self.systolic_bp = Systolic_BP
        self.diastolic_bp = Diastolic_BP
        self.heart_rate = Heart_Rate
        self.daily_steps = Daily_Steps

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Gender": [self.gender],
                "Occupation": [self.occupation],
                "Sleep_Duration": [self.sleep_duration],
                "Quality_of_Sleep": [self.quality_of_sleep],
                "Physical_Activity_Level": [self.physical_activity_level],
                "Stress_Level": [self.stress_level],
                "BMI_Category": [self.bmi_category],
                "Systolic_BP": [self.systolic_bp],
                "Diastolic_BP": [self.diastolic_bp],
                "Heart_Rate": [self.heart_rate],
                "Daily_Steps": [self.daily_steps]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)