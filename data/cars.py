import os
import json

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale as normalize_data
from sklearn.model_selection import ShuffleSplit

def read_cars_json_as_dataframe(file="cars.json", normalize=True):
    with open(file) as json_file:
        json_data = json.load(json_file)
    data = pd.DataFrame(json_data)
    # remove rows with nan
    data = data.dropna()
    # remove Name column
    data = data.drop("Name", axis=1)
    # convert Year
    year_shorter = {index: int(index[0:4]) for index in np.unique(data["Year"])}
    data["Year"] = data["Year"].replace(year_shorter)
    # convert Origin into number
    origin_number = {origin: index for index, origin in enumerate(np.unique(data["Origin"]))}
    data["Origin"] = data["Origin"].replace(origin_number)
    # change type of cylinders, year, origin to object
    new_dtype="int64"
    data["Cylinders"] = data["Cylinders"].astype(new_dtype)
    data["Year"] = data["Year"].astype(new_dtype)
    data["Origin"] = data["Origin"].astype(new_dtype)
    if normalize:
        for column in data.columns:
            if data[column].dtype == "float64":
                data[column] = normalize_data(data[column])
    return data

def train_test_validate(train_size=0.6):
    data = read_cars_json_as_dataframe()
    ss = ShuffleSplit(n_splits=1, test_size=1-train_size, random_state=42)
    train_indices, test_indices = list(ss.split(data))[0]
    ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=1337)
    test_indices, validate_indices = list(ss.split(data.iloc[test_indices]))[0]
    return data.iloc[train_indices], data.iloc[test_indices], data.iloc[validate_indices]

def train():
    train, test, validate = train_test_validate()
    return train

def test():
    train, test, validate = train_test_validate()
    return test

def validate():
    train, test, validate = train_test_validate()
    return validate

def create_file(file_name, data):
    data.to_csv(file_name, index=None)
    return file_name
