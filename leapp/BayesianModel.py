import os
import json

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from leapp.Graph import Graph, Node, DiscreteNode, ContinuousNode
from leapp.JSONModelCreator import JSONModelCreator
from leapp.JSONReader import JSONReader
from leapp.ProbParameter import is_similar, merge_nodes

class BayesianModel(object):
    def __init__(self, discrete_variables=[], continuous_variables=[],
                 blacklist_edges=None, level_dict=None, whitelist_edges=None, score="bic-cg", algo="hc"):
        self.graph = None
        self.level_dict = level_dict
        self.whitelist = whitelist_edges if whitelist_edges else []
        self.blacklist = blacklist_edges if blacklist_edges else []
        self.discrete_variables = discrete_variables
        self.continuous_variables = continuous_variables
        self.score = score
        self.algo = algo
        self.merged_parameter = 0

    def learn_through_r(self, data_file, relearn=True, install_packages=False, verbose=False):
        # if an error occurs this returns True, else False
        if relearn:
            jmc = JSONModelCreator(data_file, self.whitelist, self.discrete_variables, self.continuous_variables, self.blacklist, self.score, self.algo)
            (json_file, discrete_vars, continuous_vars) = jmc.generate_model_as_json_with_r(install_packages=install_packages, verbose=verbose)
            if json_file == None:
                bayesian_model = None
                return True
            self.discrete_variables = discrete_vars
            self.continuous_variables = continuous_vars
        else:
            json_file = data_file + ".json"
        json_reader = JSONReader(self)
        bayesian_model = json_reader.parse(json_file)
        return False

    def get_network_structure(self, verbose=False):
        bayesian_network = self.graph.get_networkx_object()
        if verbose:
            nx.draw_kamada_kawai(bayesian_network, with_labels=True)
            plt.show()

    def get_variable_categories(self):
        categories = {}
        for variable in self.discrete_variables:
            categories[variable] = list(self.get_graph().get_node(variable).get_level().values())
        return categories

    def get_variable_distributions(self):
        distributions = {}
        for variable in self.discrete_variables:
            distributions[variable] = "Binomial"
        for variable in self.continuous_variables:
            distributions[variable] = "Gaussian"
        return distributions


    def get_variable_tensors(self):
        cpt_dict = {}
        categories = self.get_variable_categories()
        for variable in self.discrete_variables:
            """
            pd.DataFrame(data={"C"  :   [False, False, False, False, True, True, True, True],
                               "B"  :   [False, False, True, True, False, False, True, True],
                               "A"  :   [False, True, False, True, False, True, False, True],
                               "Probability": [0.1, 0.3, 0.4, 0.5, 0.9, 0.7, 0.6, 0.5]}) 
            """
            # set the levels for each parent and the node
            data = {}
            # fill the conditional probability table with all data
            data = {}
            # number of inserts for each level
            number_of_inserts = 1
            # insert and unroll cases for probabilities
            for parent in self.get_graph().get_node(variable).get_discrete_parents():
                for key in data.keys():
                    data[key] = data[key]*len(categories[parent.get_name()])
                # insert each case number of inserts time
                category = []
                for case in categories[parent.get_name()]:
                    for _ in range(number_of_inserts):
                        category.append(case)
                data[parent.get_name()] = category
                number_of_inserts = number_of_inserts * len(categories[parent.get_name()])
            # do the same for the variable of interest
            for key in data.keys():
                data[key] = data[key] * len(categories[variable])
            # insert each case number of inserts time
            category = []
            for case in categories[variable]:
                for _ in range(number_of_inserts):
                    category.append(case)
            data[variable] = category
            number_of_inserts = number_of_inserts * len(categories[variable])
            # insert the flatten probability
            data["Probability"] = self.get_graph().get_node(variable).get_parameter().get_prob_tensor().ravel()
            cpt_dict[variable] = data
        return cpt_dict

    def generate_json_for_sampler(self, output_file_name):
        conditional_node_order = self.get_condition_node_order()
        bayesian_json = {}
        bayesian_json["conditional node order"] = [node.get_name() for node in conditional_node_order]
        for node in conditional_node_order:
            bayesian_json[node.get_name()] = node.get_parameter().to_json()
        with open(os.path.join(os.path.dirname(__file__), "json_files_for_sampler", output_file_name), "w") as json_file:
            json_file.write(json.dumps(bayesian_json))

    def simplify(self, tolerance, verbose=False):
        prepared_nodes = []
        most_simplify = 0
        cur_simplify = 0
        name = None
        number_of_merged_parameter = 0
        to_merge = []
        for node in self.get_graph().get_nodes():
            to_merge = []
            leafs = node.get_parameter().get_prob_graph().get_leafs()
            cur_simplify = 0
            for leaf_a in leafs:
                for leaf_b in leafs:
                    if leaf_a is not leaf_b:
                        if is_similar(tolerance, leaf_a, leaf_b):
                            to_merge.append(leaf_a)
                            to_merge.append(leaf_b)
                            #cur_simplify += 1
                            #if leaf_a not in prepared_nodes:
                            #    prepared_nodes.append(leaf_a)
                            #if leaf_b not in prepared_nodes:
                            #    prepared_nodes.append(leaf_b)
                            #merge_nodes(leaf_a, leaf_b, tolerance)
            params = 0
            mu = 0
            sd = 0
            for leaf in to_merge:
                if leaf.is_discrete():
                    params += leaf.get_parameter()
                else:
                    mu += leaf.get_parameter()[0]
                    sd += leaf.get_parameter()[1]
            if len(to_merge) > 0:
                new_param = params/len(to_merge)
                new_mean = mu/len(to_merge)
                new_sd = sd/len(to_merge)
                for leaf in to_merge:
                    if leaf.is_discrete():
                        cur_simplify += 1
                        leaf.set_parameter(new_param)
                    else:
                        cur_simplify += 2
                        leaf.set_parameter([new_mean, new_sd])
        self.merged_parameter = int(cur_simplify)

    def generate_probability_graphs(self):
        for node in self.graph.get_nodes():
            node.get_parameter().generate_graph()

    def get_graph(self):
        return self.graph

    def set_graph(self, graph):
        self.graph = graph

    def set_level_dict(self, level_dict):
        self.level_dict = level_dict

    def add_node(self, node):
        if self.graph:
            self.graph = Graph()
        self.graph.add_node(node)

    def add_edge(self, node_from, node_to):
        if self.graph.has_node(node_from) and self.graph.has_node(node_to):
            self.graph.add_edge(node_from, node_to)
        else:
            raise Exception("Please first insert nodes.")

    def add_level(self, level):
        self.level_dict = level

    def get_level_size(self, node):
        if node in self.level_dict:
            return len(self.level_dict[node])
        else:
            return 0

    def get_level_dict(self):
        return self.level_dict

    def get_level(self, node, index=None):
        if node in self.level_dict:
            if not index:
                return self.level_dict[node]
            else:
                return self.level_dict[node][index]

    def get_condition_node_order(self):
        """
        Computes the insertion order for the blog creation.
        :return: order to insert nodes
        """
        graph = self.get_graph()
        nodes = graph.get_nodes()
        inserted = []
        order = []
        while nodes:
            curr = nodes.pop(0)
            if any([parent not in inserted for parent in curr.get_parents()]):
                nodes.append(curr)
            else:
                inserted.append(curr)
                order.append(curr)
        return order

    def get_number_of_parameter(self):
        parameter_size = 0
        for nodes in self.get_graph().get_nodes():
            parameter_size += np.product(np.shape(nodes.get_parameter().get_prob_tensor()))
        return parameter_size - self.merged_parameter

    def get_graph_description(self):
        nodes = [node.get_name() for node in self.get_graph().get_nodes()]
        edges = self.get_graph().get_edges()
        graph_description = {'nodes': nodes, 'edges': edges}
        graph_description['enforced_edges'] = self.whitelist
        graph_description['forbidden_edges'] = self.blacklist
        enforced_node_dtypes = dict()
        for node in self.continuous_variables:
            enforced_node_dtypes[node] = 'numerical'
        for node in self.discrete_variables:
            enforced_node_dtypes[node] = 'string'
        graph_description['enforced_node_dtypes'] = enforced_node_dtypes
        return graph_description
