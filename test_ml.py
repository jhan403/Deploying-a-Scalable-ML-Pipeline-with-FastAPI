import pytest
# TODO: add necessary import
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
import numpy as np

test_data = pd.read_csv("data/census.csv")
train, test = train_test_split(test_data, test_size = 0.2, random_state = 42)


# TODO: implement the first test. Change the function name and input as needed
def test_model_contains_data():
    """
    # add description for the first test
    """
    assert len(test) > 0, 'Testing dataset contains no data'
    assert len(train) > 0, 'Training dataset contains no data'


# TODO: implement the second test. Change the function name and input as needed
def test_correct_model():
    """
    # add description for the second test
    """
    # Your code here
    cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"]

    X_train, y_train, encoder, lb = process_data(train, categorical_features = cat_features, 
        label = 'salary', training = True)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), 'ML model is not RandomForestClassifier.'
    

# TODO: implement the third test. Change the function name and input as needed
def test_correct_data_type():
    """
    # add description for the third test
    """
    # Your code here
    cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"]

    X_train, y_train, encoder, lb = process_data(train, categorical_features = cat_features, 
        label = 'salary', training = True)

    X_test, y_test, _, _  =  process_data(test,
        categorical_features = cat_features, label = "salary", training = False, encoder = encoder, lb = lb)
    
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), 'Data type for precision is incorrect, should be type float.'
    assert isinstance(recall, float), 'Data type for recall is incorrect, should be type float.'
    assert isinstance(fbeta, float), 'Data type for fbeta is incorrect, should be type float.'
