# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:15:45 2020

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

##########################
##### LOAD DATA  #########
##########################
features_df = pd.read_csv(
    "training_set_features.csv", 
        index_col="respondent_id"
)
labels_df = pd.read_csv(
    "training_set_labels.csv", 
        index_col="respondent_id"
)
test_features_df = pd.read_csv(
        "test_set_features.csv", 
            index_col="respondent_id"
)
submission_df = pd.read_csv("submission_format.csv",    index_col="respondent_id")
#print(test_features_df.isnull().sum())

# race 4
# sex 2
# age_group 5
# census_msa 3
# hhs_geo_region 10

def preProcess(features_df):
    #len(features_df["education"].unique().tolist()) #5
    #len(features_df["income_poverty"].unique().tolist()) #4
    #len(features_df["marital_status"].unique().tolist()) #3
    #len(features_df["rent_or_own"].unique().tolist()) #3
    #len(features_df["employment_status"].unique().tolist()) #4
    #len(features_df["household_adults"].unique().tolist()) #mean
    #len(features_df["household_children"].unique().tolist()) #mean
    #len(features_df["employment_industry"].unique().tolist()) #22
    #len(features_df["employment_occupation"].unique().tolist()) #24
    
    values = {'education': "unknown",'income_poverty': "unknown", 
              'marital_status': "unknown", 'rent_or_own': "unknown", 'employment_status': "unknown",
              'employment_industry': "unknown", 'employment_occupation': "unknown"}
    features_df = features_df.fillna(value=values)
    
    #features_df["behavioral_antiviral_meds"] = features_df['behavioral_antiviral_meds'].fillna(features_df['behavioral_antiviral_meds'].mode(), inplace=True)
    #print(features_df.isnull().sum())
    features_df = features_df.fillna(features_df.mean())
    #features_df.fillna(features_df.mode().iloc[0], inplace=True) 
    
    tem = pd.get_dummies(features_df.education, prefix='education')
    tem = tem.drop("education_unknown", axis=1)
    features_df = features_df.drop("education", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.age_group, prefix='age_group')
    tem = tem.drop("age_group_65+ Years", axis=1)
    features_df = features_df.drop("age_group", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    tem = pd.get_dummies(features_df.race, prefix='race')
    tem = tem.drop("race_White", axis=1)
    features_df = features_df.drop("race", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.sex, prefix='sex')
    tem = tem.drop("sex_Male", axis=1)
    features_df = features_df.drop("sex", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    tem = pd.get_dummies(features_df.income_poverty, prefix='income_poverty')
    tem = tem.drop("income_poverty_Below Poverty", axis=1)
    features_df = features_df.drop("income_poverty", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.marital_status, prefix='marital_status')
    tem = tem.drop("marital_status_Married", axis=1)
    features_df = features_df.drop("marital_status", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.rent_or_own, prefix='rent_or_own')
    tem = tem.drop("rent_or_own_Rent", axis=1)
    features_df = features_df.drop("rent_or_own", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.employment_status, prefix='employment_status')
    tem = tem.drop("employment_status_unknown", axis=1)
    features_df = features_df.drop("employment_status", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.hhs_geo_region, prefix='hhs_geo_region')
    tem = tem.drop("hhs_geo_region_oxchjgsf", axis=1)
    features_df = features_df.drop("hhs_geo_region", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    #features_df = features_df.drop("hhs_geo_region", axis=1)

    
    tem = pd.get_dummies(features_df.census_msa, prefix='census_msa')
    tem = tem.drop("census_msa_Non-MSA", axis=1)
    features_df = features_df.drop("census_msa", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    #features_df = features_df.drop("census_msa", axis=1)

    
    tem = pd.get_dummies(features_df.employment_industry, prefix='employment_industry')
    tem = tem.drop("employment_industry_unknown", axis=1)
    features_df = features_df.drop("employment_industry", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    
    tem = pd.get_dummies(features_df.employment_occupation, prefix='employment_occupation')
    tem = tem.drop("employment_occupation_unknown", axis=1)
    features_df = features_df.drop("employment_occupation", axis=1)
    features_df = pd.concat([features_df, tem], axis=1, sort=False)
    
    #features_df = features_df.drop("household_children", axis=1)
    #features_df = features_df.drop("household_adults", axis=1)

    #features_df = features_df.drop("h1n1_concern", axis=1)
    #features_df = features_df.drop("h1n1_knowledge", axis=1)
    #features_df = features_df.drop("doctor_recc_h1n1", axis=1)
    #features_df = features_df.drop("opinion_h1n1_vacc_effective", axis=1)
    #features_df = features_df.drop("opinion_h1n1_risk", axis=1)
    #features_df = features_df.drop("opinion_h1n1_sick_from_vacc", axis=1)
      
    return features_df


features_df = preProcess(features_df)
test_features_df = preProcess(test_features_df)


#############################
##### FEATURE SELECTION #######
#############################
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features_df = features_df.iloc[:,:]
features_df = scaler.fit_transform(features_df)


from numpy import sort
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost import XGBClassifier

class MyXGBClassifier(XGBClassifier):
	@property
	def coef_(self):
		return None
 
gbc=MyXGBClassifier(penalty="l2", objective="binary:logistic", random_state=42)

gbc=MyXGBClassifier(objective="reg:logistic", colsample_bytree=0.3, learning_rate=0.1,
                        max_depth=6, alpha=10, n_estimators= 300)

labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)

gbc.fit(features_df, labels_seasonal)

plot_importance(gbc)

thresholds = sort(gbc.feature_importances_)
thresholds
n = thresholds.shape[0]

    
selection = SelectFromModel(gbc, threshold=thresholds[3], prefit=True)
select_X_train = selection.transform(features_df)    
    





#############################
##### BUILDING MODELS #######
#############################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score

RANDOM_SEED = 6    # Set a random seed for reproducibility!

scaler = MinMaxScaler()
features_df = features_df.iloc[:,:]
features_df = scaler.fit_transform(features_df)
test_features_df = scaler.fit_transform(test_features_df)

###################################
########### Seasonal ##################
###################################
estimator=XGBClassifier(penalty="l2", objective="binary:logistic", random_state=42)

estimator=XGBClassifier(objective="reg:logistic", colsample_bytree=0.3, learning_rate=0.1,
                        max_depth=6, alpha=10, n_estimators= 300)
#labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)

X_train, X_eval, y_train, y_eval = train_test_split(
    select_X_train,
    labels_seasonal,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)
estimator.fit(X_train, y_train)
estimator.fit(features_df, labels_seasonal)

preds = estimator.predict_proba(X_eval)
preds = estimator.predict_proba(test_features_df)

preds

y_preds = pd.DataFrame(
    {
        "seasonal_vaccine": preds[:, 1],
    },
    index = y_eval.index
)
print("y_preds.shape:", y_preds.shape)
y_preds.head()

roc_auc_score(y_eval['seasonal_vaccine'], y_preds['seasonal_vaccine'])

submission_df["seasonal_vaccine"] = preds[:, 1]

###################################
########### H1N1 ##################
###################################
estimator=XGBClassifier(penalty="l2", objective="binary:logistic", random_state=42)

estimator=XGBClassifier(objective="reg:logistic", colsample_bytree=0.3, learning_rate=0.1,
                        max_depth=6, alpha=10, n_estimators= 300)
#labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)
labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_h1n1,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)
estimator.fit(X_train, y_train)

estimator.fit(features_df, labels_h1n1)

preds = estimator.predict_proba(X_eval)
preds = estimator.predict_proba(test_features_df)
preds

y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[:, 1],
    },
    index = y_eval.index
)
print("y_preds.shape:", y_preds.shape)
y_preds.head()


roc_auc_score(y_eval['h1n1_vaccine'], y_preds['h1n1_vaccine'])

submission_df["h1n1_vaccine"] = preds[:, 1]
###################################
########## Multi Label ############
###################################
features_df = scaler.fit_transform(features_df)

estimators = MultiOutputClassifier(
    estimator=XGBClassifier(penalty="l2", objective="binary:logistic", 
                            random_state=42)
)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_df,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

# Train model
estimators.fit(features_df, labels_df)

# Predict on evaluation set
# This competition wants probabilities, not labels
preds = estimators.predict_proba(X_eval)
preds
k = preds[0]
y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[0][:, 1],
        "seasonal_vaccine": preds[1][:, 1],
    },
    index = y_eval.index
)
print("y_preds.shape:", y_preds.shape)
y_preds.head()


def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(
        f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
    )
    
    
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(
    y_eval['h1n1_vaccine'], 
    y_preds['h1n1_vaccine'], 
    'h1n1_vaccine',
    ax=ax[0]
)
plot_roc(
    y_eval['seasonal_vaccine'], 
    y_preds['seasonal_vaccine'], 
    'seasonal_vaccine',
    ax=ax[1]
)
fig.tight_layout()


roc_auc_score(y_eval, y_preds)


###################################
########## Test class ############
###################################

test_probas = estimators.predict_proba(test_features_df)
test_probas

print(test_features_df.shape)
submission_df = pd.read_csv("submission_format.csv",    index_col="respondent_id")
print(submission_df.shape)
print(np.asarray(test_probas).shape)
submission_df.head()

# Save predictions to submission data frame
submission_df["h1n1_vaccine"] = test_probas[0][:, 1]
submission_df["seasonal_vaccine"] = test_probas[1][:, 1]

submission_df.head()

submission_df.to_csv('submission13.csv', index=True)