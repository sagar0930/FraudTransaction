import os
import sqlite3
import joblib
import pandas as pd
os.chdir("C://Users/Admin/Downloads/Data Science/Session_41_DS_Project_Structure/DS1/")
from data_processing_and_features import convert_dtpye, handling_blanks,handing_null,cat_feature
model = joblib.load('model_new_classifier.pkl')

conn = sqlite3.connect('Database.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
conn.close()

df = pd.read_sql_query('Select * from Fraud_detection' , conn)
data = df[5000000:]



data = pd.read_csv("data_pred.csv")
# Use same data cleaning and preprocessing pipeline as testing
data = convert_dtpye(data)
data = handling_blanks(data)
data = handing_null(data)
data = cat_feature(data)

results_label = model.predict(data)
results_probability = model.predict_proba(data)