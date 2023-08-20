import numbers

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class DataTransformer(object):
    def __init__(self, numeric=False, save=True):
        self.datamap = None
        self.data = None
        self.numeric = numeric
        self.save = save
        self.discrete_variables = None

    def clean(self, empty_signs = None):
        if not empty_signs:
            empty_signs = ["?"]
        # remove the non complete rows
        for column in self.data.columns:
            for symbol in empty_signs:
                if symbol in np.unique(self.data[column]):
                    self.data = self.data[self.data[column] != '?']
        pass

    def calculate_discrete_variables(self, percentage=99):
        if self.data is None:
            return []
        else:
            discrete_variables = []
            for var in self.data.columns:
                if not all([isinstance(i, numbers.Number) for i in self.data[var]]):
                    #if pd.api.types.is_string_dtype(self.data[var]):
                    discrete_variables.append(var)
                else:
                    if len(np.unique(self.data[var])) / len(self.data[var]) * 100 > percentage and self.numeric:
                        discrete_variables.append(var)
        self.discrete_variables = discrete_variables
        return discrete_variables

    def get_discrete_variables(self):
        return self.discrete_variables

    def transform(self, file_name, ending_comma=False, index_column=False, discrete_variables=None):
        """
        ending_comma
            True if after each row in the csv there is a comma without data
        index_column
            True if there is a column with indices, we omit these
        """
        self.read_csv(file_name, index_column, ending_comma)
        self.clean()
        # self.to_string_cols(df=self.data, columns=discrete_variables, inplace=True)
        self.datamap = DataMap()
        if discrete_variables is None:
            discrete_variables = self.calculate_discrete_variables()
        self.datamap.generate_map(self.data[discrete_variables])
        for column in discrete_variables:
            self.data[column] = pd.Series(self.data[column]).map(self.datamap.get_map(column))
        if self.save:
            return self.save_csv(file_name)


    def to_string_cols(self, df, columns=None, inplace=False):
        """Replace columns with their string representation."""
        if columns is None:
            columns = df.columns
        if not inplace:
            df = df.copy()
        df.loc[:, columns] = df.loc[:, columns].applymap(str)
        return df

    def read_csv(self, file_name, index_column, ending_comma=False):
        self.data = pd.read_csv(file_name, na_filter=False, index_col=0 if index_column else False)
        if ending_comma:
            self.data = self.data.drop(columns=self.data.columns[len(self.data.columns)-1])

    def save_csv(self, file_name):
        self.data.to_csv(file_name[:-4]+"_cleaned.csv", index=False)
        return file_name[:-4]+"_cleaned.csv"

    def get_map(self):
        return self.datamap


class DataMap(object):
    def __init__(self):
        self.map = dict()

    def generate_map(self, data):
        for column in data:
            values = np.unique(data[column])
            values_to_int = {i: value for i, value in enumerate(values)}
            self.map[column] = values_to_int

    def get_map(self, column, reverse=True):
        if reverse:
            return {value: key for key, value in self.map[column].items()}
        else:
            return self.map[column]

    def __repr__(self):
        repr = ""
        for key, value in self.map.items():
            repr += f"{key}: {value}\n"
        return repr

if __name__ == "__main__":
    pass

    file = "/home/julien/PycharmProjects/util/bayesian_network_learning/data/allbus.csv"
    dt = DataTransformer()
    dt.transform(file, discrete_variables=["sex","educ","eastwest","happiness","health","lived_abroad","spectrum"], index_column=True)

    print(dt.get_map())
