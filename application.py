import os

from flask import Flask, request, render_template

# from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import generate_prediction_text

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Age=int(request.form.get('age')),
            Gender=request.form.get('gender'),
            Occupation=request.form.get('occupation'),
            Sleep_Duration=float(request.form.get('sleep_duration')),
            Quality_of_Sleep=float(request.form.get('quality_of_sleep')),
            Physical_Activity_Level=float(request.form.get('physical_activity_level')),
            Stress_Level=float(request.form.get('stress_level')),
            BMI_Category=request.form.get('bmi_category'),
            Systolic_BP=int(request.form.get('systolic_bp')),
            Diastolic_BP=int(request.form.get('diastolic_bp')),
            Heart_Rate=int(request.form.get('heart_rate')),
            Daily_Steps=int(request.form.get('daily_steps'))
        )
        pred_df = data.get_data_as_dataframe()
        # print(pred_df)
        
        pred_pipeline = PredictPipeline()
        results = pred_pipeline.predict(pred_df)
        
        result = results[0]
        heading, explanation, advice = generate_prediction_text(result)
        
        return render_template(
            'result.html', 
            heading=heading, 
            explanation=explanation, 
            advice=advice)
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))