# Learn Probabilistic Programs (leapp)

## Install and Requirements
Create a new conda environment with

    conda env create -f environment.yml
    
and activate it with

    conda activate leapp
    
delete it with

    conda env remove -n leapp

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

### Example
Try the run the `example.py` file. This creates a simple PyMC3 code fragment for the cars data set. 


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

#### 3. `OSError` with R
`OSError: cannot load library 'C:\Program Files\R\R-3.6.1\bin\x64\R.dll'`

**Solution** Please set the path variables for your R.dll. Especially R_HOME and PATH or try adding to your code.
```
import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-3.6.1"
os.environ["PATH"]   = r"C:\Program Files\R\R-3.6.1\bin\x64" + ";" + os.environ["PATH"]
```