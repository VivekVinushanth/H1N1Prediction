"""Class to compare performance with different classifiers"""
import sys

sys.path.append('../../')
# sys.path.append('/content/Modified-Geometric-Smote/')
import numpy as np
import pandas as pd
import xgboost as xgb
# import applications.main as gsom
from classifier.preprocessing import pre_process as pp, getTest

#  Directory
path = '../../data/'

# def getTest(filename):
#     df = pd.read_csv(filename)
#
#     X = np.asarray(df.iloc[1:-1].values)
#
#
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     return X, index

def evaluate(classifier,y_predict,index):
    # Write output to csv file
    print(classifier)

    print(index.shape)
    index.resize(26708,1)
    print(y_predict.shape)
    data = np.column_stack([index,y_predict])
    label= ["respondent_id","seasonal_vaccine"]
    frame  = pd.DataFrame(data,columns=label)
    export_csv = frame.to_csv(r'output/seasonal_vaccine.csv',header=True)



def XGBoost(X_train,y_train,X_test):

    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict_proba(X_test)
    y_predict = y_predict[:,1]
    return evaluate("XGBoost", y_predict, index)




# data transformation if necessary.
X_t, y_t = pp('training_set_seasonal.csv')

# For Non-resampled
X_train,y_train = X_t,y_t

X_test,index = getTest('test_set_features.csv')

# For SMOTE
# sm = SMOTE()
# X_train, y_train = sm.fit_resample(X_t, y_t)

# GSMOTE = EGSmote()
# # GSMOTE = OldGeometricSMOTE()
# X_train, y_train = GSMOTE.fit_resample(X_t, y_t)

XGBoost(X_train,y_train,X_test)


