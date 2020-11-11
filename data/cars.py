import os
import json

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale as normalize_data

# from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical

def read_cars_json_as_dataframe(file=os.path.join("data","cars.json"), normalize=False):
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

def create_file(file_name, data):
    data.to_csv(file_name, index=None)
    return file_name

"""
def get_spn_types(with_name=True):
    spn_parameter_types = {
        'Miles_per_Gallon': Gaussian,
        'Cylinders': Categorical,
        'Displacement': Gaussian,
        'Horsepower': Gaussian,
        'Weight_in_lbs': Gaussian,
        'Acceleration': Gaussian,
        'Year': Categorical,
        'Origin': Categorical
    }
    if with_name:
        return spn_parameter_types
    return list(spn_parameter_types.values())
"""