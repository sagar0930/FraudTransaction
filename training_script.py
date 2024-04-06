import joblib
import sqlite3
import pandas as pd
from data_processing_and_features import convert_dtpye, handling_blanks,handing_null,cat_feature
from model_building import train_test_split, imbalanced_data, fit_and_evaluate_model,  get_important_features
import os
os.chdir("C://Users/Admin/Downloads/Data Science/Session_41_DS_Project_Structure/DS1/")

conn = sqlite3.connect('Database.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
conn.close()
conn = sqlite3.connect('Database.db')
df = pd.read_sql_query('Select * from Fraud_detection' , conn)

df = convert_dtpye(df)
df = handling_blanks(df)
df = handing_null(df)
df = cat_feature(df) 

train_data = df[:4000000]
test_data = df[4000000:5000000]
prod_eval_data = df[5000000:]

x_train, x_test, y_train, y_test, features = train_test_split(train_data,
                                                    test_data)

x_res, y_res = imbalanced_data(x_train,y_train)

model = fit_and_evaluate_model(x_res, x_test, y_res, y_test)

feature_importance = get_important_features(model, features)


joblib.dump(model , 'model_new_classifier.pkl')

