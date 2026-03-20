{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4c7440db-1a7e-443f-91e2-65edef75fd0d",
   "metadata": {},
   "source": [
    "# 💤 Sleep-Disorder-Prediction\n",
    "A data-driven analysis exploring the relationship between lifestyle and sleep disorder.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99b32cac-59aa-4cf8-aa0e-9113d9265eba",
   "metadata": {},
   "source": [
    "## Project Objective:\n",
    "The focus of this project is to predict sleep disorders from lifestyle factors and cardiovascular functions.\n",
    "The factors include age, gender, sleep quality, physical activity level, stress level, occupation, heart rate, blood pressure and BMI.\n",
    "The dataset for this project was obtained from Kaggle, and it contains 374 rows and 13 columns, with 11 independent variables (excluding Person ID) and 1 target variable, 'Sleep Disorder'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d6e6926-02fe-4ce0-a11d-4cc8f5a177e3",
   "metadata": {},
   "source": [
    "## Dataset Description:\n",
    "The dataset provides a detailed compilation of sleep disorder and lifestyle factors.\n",
    "The dataset includes the following key features;\n",
    "Comprehensive Sleep Metrics: This section provides details of sleep duration, sleep quality and the factors influencing sleep patterns.\n",
    "Lifestyle Factors: This section provides information about age, gender, physical activity levels, stress levels, and BMI categories.\n",
    "Cardiovascular Health: This section provides information about blood pressure and heart rate measurements.\n",
    "Sleep Disorder Analysis: This section identifies the occurrence of sleep disorders with the following categories;\n",
    "- None: Individuals with no sleep disorder\n",
    "- Insomnia: Individuals that experience difficulty falling asleep or staying asleep, leading to inadequate or poor sleep quality.\n",
    "- Sleep Apnea: Individuals who suffer from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "457123a1-586a-45d5-9740-0b013038e1e5",
   "metadata": {},
   "source": [
    "## Dataset Information:\n",
    "| Feature                       | Description                                                                |\n",
    "|-------------------------------|-----------------------------------------------------------------------------|\n",
    "| Person ID                     | An identifier for each individual.                                          |\n",
    "| Gender                        | The gender of the person (Male/Female).                                     |\n",
    "| Age                           | The age of the person in years.                                             |\n",
    "| Occupation                    | The occupation of the person.                                               |\n",
    "| Sleep Duration (hours)        | The number of hours the person sleeps per day.                              |\n",
    "| Quality of Sleep (1-10)       | A subjective rating of sleep quality (scale of 1–10).                       |\n",
    "| Physical Activity (min/day)   | Minutes of daily physical activity.                                         |\n",
    "| Stress Level (1-10)           | A subjective rating of stress level (scale of 1–10).                        |\n",
    "| BMI Category                  | BMI classification (e.g., underweight, normal, overweight).                 |\n",
    "| Blood Pressure (sys/dia)      | Systolic/Diastolic blood pressure reading.                                  |\n",
    "| Heart Rate (bpm)              | Resting heart rate in beats per minute.                                     |\n",
    "| Daily Steps                   | Number of steps taken per day.                                              |\n",
    "| Sleep Disorder                | Presence of a sleep disorder (None, Insomnia, Sleep Apnea).                 |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1ae1607-3ab1-4e92-8a7a-176e0a99dc2c",
   "metadata": {},
   "source": [
    "## Tools and Libraries used\n",
    "- Python\n",
    "- Pandas \n",
    "- Numpy\n",
    "- Matplotlib \n",
    "- Seaborn \n",
    "- Scikit-learn \n",
    "- Jupyter Notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41ded2a1-a590-411b-9773-e0305b9bdb26",
   "metadata": {},
   "source": [
    "## Steps Taken\n",
    "- Data Importation\n",
    "- Data Cleaning\n",
    "- Exploratory Data Analysis\n",
    "- Data Preprocessing\n",
    "- Model development\n",
    "- Model Evaluation\n",
    "- Model Refinement (Feature Selection)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "797c949c-c8d6-4b6c-bb38-dff9de51a7d1",
   "metadata": {},
   "source": [
    "## Key Visualizations\n",
    "### 🧠 Lifestyle vs Sleep Disorder\n",
    "![Lifestyle vs Sleep Disorder](images/Lifestyle-vs-Sleep-Disorders.png)\n",
    "\n",
    "### Cardiovascular Function vs Sleep Disorder\n",
    "![Cardiovascular Function vs Sleep Disorder](images/Cardiovascular-Functions-vs-Sleep-Disorders.png)\n",
    "\n",
    "*More visuals in the notebook.*"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b1cbef4-cf7a-4819-bb4e-71ee25f7147e",
   "metadata": {},
   "source": [
    "## 📌 Key Insights\n",
    "- Sleep apnea appeared to be more prevalent in older adults\n",
    "- Sleep disorders were seen more in individuals who are overweight and obese\n",
    "- Sleep disorders were seen more in nurses, teachers and salespersons\n",
    "- Sleep disorders were predominantly high in individuals with high stress level"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f58cc864-021e-4719-923b-ed072d0c2488",
   "metadata": {},
   "source": [
    "## Model Building\n",
    "Classification models were used;\n",
    "- Random Forest\n",
    "- Decision Trees\n",
    "- Extra Trees\n",
    "- Support Vector Machine\n",
    "\n",
    "##### Performance Metrics\n",
    "Random Forest and Support Vector Machine had the best performance\n",
    "</br>\n",
    "##### Random Forest\n",
    "+ Accuracy score: 0.867\n",
    "+ Precision: 0.872\n",
    "+ Recall: 0.867\n",
    "+ F1 score: 0.858\n",
    "+ CV mean: 0.906\n",
    "+ SV std: 0.039\n",
    "</br>\n",
    "</br>\n",
    "##### Support Vector Classifier\n",
    "+ Accuracy score: 0.867\n",
    "+ Precision: 0.868\n",
    "+ Recall: 0.867\n",
    "+ F1 score: 0.867\n",
    "+ CV mean: 0.893\n",
    "+ SV std: 0.041\n",
    "</br>\n",
    "However, while these two models have very similar performance, a major difference between both is that SVC performs significantly better in predicting Class 1 (Sleep Apnea) with a precision and recall score of 0.81. Random Forest on the other hand performas poorly in predicting Class 1 (Sleep Apnea) with a precision and recall scores of 0.90 and 0.56 respectively.\n",
    "</br>SVC's cross validation mean is also closer to it's accuracy score than RF's own.\n",
    "</br>As a result of all these, Support Vector Classifier is the best model of choice!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d75f5ad-2664-47f8-9439-fc20bac34b89",
   "metadata": {},
   "source": [
    "## Project Impact\n",
    "This project will provide insight into the factors influencing sleep disorders, and it will develop a model to predict and identify individuals who are at risk for sleep disorder.\n",
    "This will improve overall health among all populations"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
