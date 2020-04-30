"""Class to pre-process the input data"""

import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def pre_process(filename):
    path = "data/"
    df = pd.read_csv(path+filename)

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

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X,y

def getTest(filename):
    filename = "data/"+filename
    df = pd.read_csv(filename)

    # Fill the NaN with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # df = df.drop_duplicates(df.iloc[:, :-1].columns, keep='last')

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


    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X,index
