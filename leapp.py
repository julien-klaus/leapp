from pathlib import Path
import os
import argparse

import pandas as pd

from leapp import LearnPP

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn Probabilistic Programs")
    parser.add_argument('file', metavar='file', type=str, help="csv file with data")

    lp = LearnPP()

    args = parser.parse_args()

    data = pd.read_csv(args.file)

    lp.fit(data)

    print(lp.get_pymc_code())





