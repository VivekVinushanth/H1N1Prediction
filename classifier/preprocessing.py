"""Class to pre-process the input data"""

import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def pre_process(filename):
    path = "data/"
    df = pd.read_csv(path + filename)

    # Fill the NaN with mean
    # df.fillna(df.mean(), inplace=True)

    # Fill the NaN with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [21])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [25])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [28])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [31])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [32])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [34])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [35])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [36])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [38])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [47])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [51])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [71])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y


def getTest(filename):
    filename = "data/" + filename
    df = pd.read_csv(filename)

    # Fill the NaN with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    X = df.iloc[:, 1:].values
    index = np.asarray(df.iloc[:, 0].values)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [21])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [25])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [28])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [31])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [32])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [34])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [35])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [36])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [38])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [47])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [51])], remainder='passthrough')
    # X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    #
    # columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [71])], remainder='passthrough')
    # X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, index


def pre_process_chain(filename):
    path = "data/"
    df = pd.read_csv(path + filename)

    # Fill the NaN with mean
    # df.fillna(df.mean(), inplace=True)

    # Fill the NaN with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1:-3].values

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [21])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [25])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [28])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [31])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [32])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [34])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [35])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [36])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [38])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [47])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print("pre-processing done")

    return X, y


def pre_process2(filename):
    path = "data/"
    df = pd.read_csv(path + filename)

    # Fill the NaN with mean
    # df.fillna(df.mean(), inplace=True)

    # Fill the NaN with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(drop='first'), [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30])],
        remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def getTest(filename):
    filename = "data/" + filename
    df = pd.read_csv(filename)

    # Fill the NaN with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    X = df.iloc[:, 1:].values
    index = np.asarray(df.iloc[:, 0].values)

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(drop='first'), [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30])],
        remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, index


def get_not_all_encode(filename):
    filename = "../data/" + filename
    print(filename)
    df = pd.read_csv(filename)

    # Fill the NaN with mode
    df["age_group"].fillna("unknown1", inplace=True)
    df["education"].fillna("unknown2", inplace=True)
    df["race"].fillna("unknown3", inplace=True)
    df["sex"].fillna("unknown4", inplace=True)
    df["income_poverty"].fillna("unknown5", inplace=True)
    df["marital_status"].fillna("unknown6", inplace=True)
    df["rent_or_own"].fillna("unknown7", inplace=True)
    df["employment_status"].fillna("unknown8", inplace=True)
    df["hhs_geo_region"].fillna("unknown9", inplace=True)
    df["census_msa"].fillna("unknown10", inplace=True)
    # df["employment_industry"].fillna("unknown11", inplace=True)
    # df["employment_occupation"].fillna("unknown12", inplace=True)

    df.fillna(df.mode().iloc[0], inplace=True)
    # print(df.head(15))
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(drop='first'), [21, 22, 23, 24, 25, 26, 27, 28, 29, 30])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def get_not_all_encode_test(filename):
    filename = "../data/" + filename
    df = pd.read_csv(filename)

    # Fill the NaN with mode
    # df.fillna(df.mode().iloc[0], inplace=True)
    # df.fillna({'a': 0, 'b': 0}, inplace=True)

    df["age_group"].fillna("unknown1", inplace=True)
    df["education"].fillna("unknown2", inplace=True)
    df["race"].fillna("unknown3", inplace=True)
    df["sex"].fillna("unknown4", inplace=True)
    df["income_poverty"].fillna("unknown5", inplace=True)
    df["marital_status"].fillna("unknown6", inplace=True)
    df["rent_or_own"].fillna("unknown7", inplace=True)
    df["employment_status"].fillna("unknown8", inplace=True)
    df["hhs_geo_region"].fillna("unknown9", inplace=True)
    df["census_msa"].fillna("unknown10", inplace=True)
    # df["employment_industry"].fillna("unknown11", inplace=True)
    # df["employment_occupation"].fillna("unknown12", inplace=True)

    df.fillna(df.mode().iloc[0], inplace=True)

    # print(df.head(15))

    X = df.iloc[:, 1:].values
    index = np.asarray(df.iloc[:, 0].values)

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(drop='first'), [21, 22, 23, 24, 25, 26, 27, 28, 29, 30])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, index


def feature_selection(filename):
    filename = "../data/" + filename
    df = pd.read_csv(filename)
    features_df = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values

    df["age_group"].fillna("unknown1", inplace=True)
    df["education"].fillna("unknown2", inplace=True)
    df["race"].fillna("unknown3", inplace=True)
    df["sex"].fillna("unknown4", inplace=True)
    df["income_poverty"].fillna("unknown5", inplace=True)
    df["marital_status"].fillna("unknown6", inplace=True)
    df["rent_or_own"].fillna("unknown7", inplace=True)
    df["employment_status"].fillna("unknown8", inplace=True)
    df["hhs_geo_region"].fillna("unknown9", inplace=True)
    df["census_msa"].fillna("unknown10", inplace=True)
    # df["employment_industry"].fillna("unknown11", inplace=True)
    # df["employment_occupation"].fillna("unknown12", inplace=True)

    df.fillna(df.mode().iloc[0], inplace=True)


    features_df = df.iloc[:, 1:].values

    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(drop='first'), [21, 22, 23, 24, 25, 26, 27, 28, 29, 30])], remainder='passthrough')

    features_df = np.array(columnTransformer.fit_transform(features_df), dtype=np.str)

    scaler = StandardScaler()
    features_df = scaler.fit_transform(features_df)

    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(features_df, labels)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(features_df.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(35, 'Score'))  # print 10 best features

# feature_selection("train_h1n1.csv")