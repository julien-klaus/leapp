# import os
# os.environ["R_HOME"] = r"C:\Program Files\R\R-3.6.1"
# os.environ["PATH"]   = r"C:\Program Files\R\R-3.6.1\bin\x64" + ";" + os.environ["PATH"]

from data import cars

from leapp.LearnPP import LearnPP

from leapp import LearnPP

if __name__ == "__main__":

    data = cars.read_cars_json_as_dataframe()

    from sklearn.datasets import load_iris
    data = load_iris()
    print(data.data)
    import numpy as np
    print([data.data.T, data.target])

    # lp = LearnPP(verbose=True)
    # lp.fit(data)

    # print(lp.get_pymc_code())