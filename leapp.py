from pathlib import Path
import os
import argparse

import pandas as pd

from leapp import LearnPP

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn Probabilistic Programs")
    parser.add_argument('--input', "-i", type=str, help="input csv file")
    parser.add_argument('--blog', action="store_true", help="return blog code", default=True)
    parser.add_argument('--pymc', action="store_true", help="return pymc codecode")

    lp = LearnPP()

    args = parser.parse_args()

    data = pd.read_csv(args.input)

    lp.fit(data)
    print()
    if args.pymc:
        print("# PYMC CODE")
        print(lp.get_pymc_code())
    if args.blog:
        print("# BLOG CODE")
        print(lp.get_blog_code())






