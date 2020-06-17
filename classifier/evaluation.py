"""Class to compare performance with different classifiers"""
import sys

from matplotlib import pyplot
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance

sys.path.append('../../')
# sys.path.append('/content/Modified-Geometric-Smote/')
import numpy as np
import pandas as pd
import xgboost as xgb
# import applications.main as gsom
from classifier.preprocessing import pre_process as pp, getTest, pre_process_chain as ppc, pre_process2 as pp2
from classifier.preprocessing import get_not_all_encode as pp_not_encode, get_not_all_encode_test as pp_test_not_encode
from sklearn.multioutput import ClassifierChain

#  Directory
path = '../../data/'


def evaluate(classifier, y_predict, index):
    # Write output to csv file
    index.resize(26708, 1)
    data = np.column_stack([index, y_predict])
    label = ["respondent_id", "h1n1_vaccine"]
    frame = pd.DataFrame(data, columns=label)
    export_csv = frame.to_csv(r'../output/h1n1_feature.csv', header=True)


def XGBoost(X_train, y_train, X_test):
    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict_proba(X_test)
    y_predict = y_predict[:, 1]
    return evaluate("XGBoost", y_predict, index)

# define custom class to fix bug in xgboost 1.0.2
class MyXGBClassifier(xgb.XGBClassifier):
	@property
	def coef_(self):
		return None

def XGBoost_feature_selection(X_train, y_train, X_test):

    # Fitting X-Gradient boosting
    gbc = MyXGBClassifier(objective="binary:logistic", random_state=42,importance_type="gain")
    gbc.fit(X_train, y_train)
    # plot feature importance
    plot_importance(gbc)
    pyplot.show()

    # # Fit model using each importance as a threshold
    thresholds = sort(gbc.feature_importances_)
    
    print (thresholds.shape[0])
    print(thresholds)
    thresh=thresholds[-10]
    print(thresh)
    selection = SelectFromModel(gbc, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    print(select_X_train.shape[1])

    # train model
    selection_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)

    # Predicting the Test set results
    y_predict = selection_model.predict_proba(select_X_test)
    y_predict = y_predict[:, 1]

    return evaluate("XGBoost", y_predict, index)



def XGBoostChain(X_train, y_train, X_test):
    print("fitting the data")
    # Fitting X-Gradient boosting
    gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    chains = [ClassifierChain(gbc, order='random', random_state=i) for i in range(10)]

    for chain in chains:
        chain.fit(X_train, y_train)

    Y_pred_chains = np.array([chain.predict_proba(X_test) for chain in chains])
    Y_pred_ensemble = Y_pred_chains.mean(axis=0)
    print(Y_pred_ensemble)
    # y_predict = y_predict[:,1]
    # return evaluate("XGBoost", y_predict, index)


# data transformation if necessary.
# X_t, y_t = pp('training_set_h1n1.csv')

# Pre-process for chain
# X_t, y_t = pp2('train_h1n1.csv')
X_t, y_t = pp_not_encode('train_h1n1.csv')

# For Non-resampled
X_train, y_train = X_t, y_t

X_test, index = pp_test_not_encode('test.csv')
# X_test,index = getTest('test_set_features_chain.csv')

# XGBoost(X_train, y_train, X_test)
XGBoost_feature_selection(X_train, y_train, X_test)

# Chain of classifiers
# XGBoostChain(X_train,y_train,X_test)
