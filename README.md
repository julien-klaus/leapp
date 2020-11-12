# LEArn Probabilistic Programs (leapp)

## Install and Requirements
Please install the package using `pip install .`

We need the following requirements for Python (3.7):

- `networkx` - for graphical output of the underlying networks
- `graphviz` - for graphical out of the underlying networks
- `pandas` - for reading and writing data
- `rpy2` - for running R code 
- `numpy` - for some mathematical calculations

Further we need an R (3.6.1) installation with the following packages:

- `bnlearn` - for learning a bayesian network
- `jsonlite` - for creating and writing a json description of the network
- (`Rgraphviz` - for plotting the network structure if some want to (see R-code))

## Usage

### Terminal
\>\>\> `python leapp.py` _csv\_file_

### Python
After installing the package one can learn the probabilistic programm. 
```
from leapp import LearnPP

lp = LearnPP()
lp.fit(data)
```
- `data` has to be a Pandas `DataFrame` object.
- `LearnPP` accepts the following parameter
    - `continuous_variables` - variables that are continuous (list) 
    - `discrete_variables` - varaibles that are discrete (list)
    - `whitelist_edges` - edges that must be in the model (list of tuples)
    - `blacklist_edges` - edges that are not allowed in the model (list of tuples)
    - `score` - score for the structure search, default `bic` (string)
    - `algo` - algorithm for the structure search, default `hc` (string)
    - `simplify_tolerance` - tolerance to merge similar distributions (float)
    - `verbose` - see more detail (boolean)
- `fit` accepts the following additional parameter
    - `transform_data` - if strings in the data frame we can replace them by numbers
    - `cleanup` - if there are `?` in the data frame we can remove them (data frame has to be complete)

## Troubleshooting
#### 1. Error during learning 
`Error in [[<-.data.frame(*tmp*, var, value = numeric(0)) : replacement has 0 rows ...`

The data object has an error. Maybe there is an index column or there are problematic column names. 

**Solution** Check the data frame or look into the R code and print there the `data` variable to see if there are some missmatches.

#### 2. PyMC contains NaN

**Solution** We encounter this with the `loglik` score. May another score can fix this. 