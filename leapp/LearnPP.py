import os

import pandas as pd
import numpy as np

from leapp.BlogCreator import BlogCreator
from leapp.PyMCCreator import PyMCCreator
from leapp.BayesianModel import BayesianModel
from leapp.DataTransformer import DataTransformer


class LearnPP():

    def __init__(self, model_name="", continuous_variables=None, discrete_variables=None, whitelist_edges=None,
                 blacklist_edges=None, score="aic-cg", algo="hc", simplify_tolerance=0.0, verbose=False,
                 install_packages=False):
        self.model_name = model_name
        self.discrete_vars = discrete_variables if discrete_variables else []
        self.continuous_vars = continuous_variables if continuous_variables else []
        self.edges_white = whitelist_edges if whitelist_edges else []
        self.edges_black = blacklist_edges if blacklist_edges else []
        self.score = score
        self.simplify_tolerance = simplify_tolerance
        self.algo = algo
        self.verbose = verbose
        self.install_packages = install_packages
        self.map = None
        self.bayesian_model = None

    def fit(self, X, transform_data=True, cleanup=False):
        assert isinstance(X, pd.DataFrame), "X has to be a Pandas DataFrame object."
        # first we need the data as a csv file
        file_name = self._save_tmp_data(X)

        # if there are not discrete variables, use the heuristic
        if len(self.discrete_vars) == 0:
            self.discrete_vars = self._get_discrete_variables(X)

        if not X.applymap(lambda x: isinstance(x, (int, float))).all().all():
            transform_data = True

        # we can only deal with numbers, so we should transform all data
        if transform_data:
            dt = DataTransformer()
            file_cleaned = dt.transform(file_name, ending_comma=False, discrete_variables=self.discrete_vars)
            self.map = dt.get_map()
            print("You have categorical data with characters in you code. I had to transform it into numerical values. Please note the following mapping:", self.map, sep="\n")
        else:
            file_cleaned = file_name

        # now we can learn the bayesian model with bnlearn
        bayesian_model = BayesianModel(continuous_variables=self.continuous_vars, discrete_variables=self.discrete_vars,
                                       whitelist_edges=self.edges_white,
                                       blacklist_edges=self.edges_black, score=self.score, algo=self.algo)
        bayesian_model.learn_through_r(file_cleaned, relearn=True, verbose=self.verbose, install_packages=self.install_packages)

        # and merge similar distributions given the simplify tolererance
        if self.simplify_tolerance != 0.0:
            bayesian_model.simplify(self.simplify_tolerance)

        self.bayesian_model = bayesian_model

        # clean up created files
        if cleanup:
            os.remove(file_cleaned)
            os.remove(file_name)
            os.remove(file_cleaned+".json")

    def _get_discrete_variables(self, data):
        discrete_vars = []
        for column, dtype in zip(data.columns, data.dtypes):
            if dtype == 'object':
                discrete_vars.append(column)
            if dtype == 'int64':
                if len(np.unique(data[column])) <= 20:
                    discrete_vars.append(column)
        return discrete_vars

    def _save_tmp_data(self, X, file_name="_tmp.dat"):
        X.to_csv(file_name, index=None)
        return os.path.abspath(file_name)

    def export_graph(self, filename="graph.pdf", view=True):
        return self.bayesian_model.get_graph().export_as_graphviz(filename, view)

    def get_pymc_code(self, function_name="trace", with_model=True, environments=True,
                      as_function=False, with_trace=False, number_of_samples=10000, query=None):
        pymc = PyMCCreator(self.bayesian_model)
        return pymc.generate(function_name=function_name, with_model=with_model,
                             as_function=as_function, with_trace=with_trace,
                             number_of_samples=number_of_samples, query=query, environments=environments)

    def get_blog_code(self):
        blog = BlogCreator(self.bayesian_model)
        return blog.generate()

    def _get_description(self):

        descr = self.bayesian_model.get_graph_description()

        cond_independence_graph = self.bayesian_model.get_network_structure(verbose=False)
        variable_categories = self.bayesian_model.get_variable_categories()
        cpt_dict = self.bayesian_model.get_variable_tensors()

        dist_dict = self.bayesian_model.get_variable_distributions()

        conditional_node_order = self.bayesian_model.get_condition_node_order()
        bayesian_json = {}
        bayesian_json["conditional node order"] = [node.get_name() for node in conditional_node_order]

        for node in conditional_node_order:
            # print_prob_table(node.get_parameter().get_prob_graph(), file_name=node.get_name(), view=False, simple=True)
            bayesian_json[node.get_name()] = node.get_parameter().to_json()
