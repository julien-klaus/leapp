# import os
# os.environ["R_HOME"] = r"C:\Program Files\R\R-3.6.1"
# os.environ["PATH"]   = r"C:\Program Files\R\R-3.6.1\bin\x64" + ";" + os.environ["PATH"]

from data import cars

from leapp import LearnPP

if __name__ == "__main__":

    data = cars.read_cars_json_as_dataframe()

    lp = LearnPP(verbose=False)
    lp.fit(data)

    print(lp.get_pymc_code())