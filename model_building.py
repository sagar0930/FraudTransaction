import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

def train_test_split(train_data,test_data):
    x_train = train_data.drop('isFraud',axis=1)
    y_train = train_data['isFraud']
    x_test = test_data.drop('isFraud',axis=1)
    y_test = test_data['isFraud']
    features = list(x_train.columns)
    return x_train, x_test, y_train, y_test, features

def imbalanced_data(x_train,y_train):
    sampling_strategy = 0.3  # Define the percentage of oversampling 
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)# Apply SMOTE with specified oversampling percentage
    x_res, y_res = smote.fit_resample(x_train, y_train)
    return x_res, y_res

def fit_and_evaluate_model(x_res, x_test, y_res, y_test,max_depth=5,min_samples_split=0.01,max_features=0.8,max_samples=0.8):
    random_forest =  RandomForestClassifier(random_state=0,\
                                            max_depth=max_depth,\
                                            min_samples_split=min_samples_split,\
                                            max_features=max_features,
                                            max_samples=max_samples, class_weight= {0:1,1:50})

    model = random_forest.fit(x_res, y_res)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("confussion matrix")
    print(random_forest_conf_matrix)
    print("\n")
    print("Accuracy of Random Forest:",random_forest_acc_score*100,'\n')
    print(classification_report(y_test,random_forest_predict))
    return model

def get_important_features(model, features):
    importances = pd.DataFrame(model.feature_importances_)
    importances['features'] = features
    importances.columns = ['importance','feature']
    importances.sort_values(by = 'importance', ascending= False,inplace=True)
    return importances
