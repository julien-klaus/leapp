import os

import itertools
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse
import jinja2


class PyMCCreator(object):
    TEMPLATE_FILE = "pymc.py.jinja"

    def __init__(self, bayesian_model):
        self.bm = bayesian_model
        self.level_dict = self.bm.get_level_dict()
        self.template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(os.path.dirname(__file__), "templates"))
        self.template_env = jinja2.Environment(loader=self.template_loader, trim_blocks=True, lstrip_blocks=True)
        self.code_template = self.template_env.get_template(self.TEMPLATE_FILE)

    def generate(self, function_name="trace", with_model=True, environments=True, as_function=False, with_trace=False, number_of_samples=1000, query=None):
        render_context = {"environments": environments,
                          "as_function": as_function,
                          "with_trace": with_trace,
                          "number_of_samples": number_of_samples,
                          "query": query,
                          "generate_code_for": self._generate_code_for,
                          "insertion_order": self.bm.get_condition_node_order()}
        outputText = self.code_template.render(render_context)
        return outputText

    def generate_and_save_code(self, file_name, function_name="trace", with_model=True, as_function=True,
                               with_trace=True, number_of_samples=1000):
        code = self.generate(function_name, with_model, as_function, with_trace, number_of_samples)
        with open(file_name, "w") as file:
            file.write(code)


    def _generate_code_for(self, node, tree, case, query=None):
        name = node.get_name()
        code = ""
        number_of_children = len(tree.get_children())
        switch = False
        for index, child in enumerate(tree.get_children()):
            # The childs are the leafs
            if child.get_name() == name:
                if case == "prob":
                    child_parameter = []
                    for child in tree.get_children():
                        child_parameter.append(child.get_parameter())
                    #code += "[" + ",".join([str(parameter) for parameter in tree.get_parameter()]) + "]"
                    code += "[" + ",".join([str(p) for p in child_parameter]) + "]"
                    break
                elif case == "mu":
                    child = tree.get_children()[0]
                    code += str(child.get_parameter(0))
                elif case == "sigma":
                    code += str(child.get_parameter(1))
            else:
                # discrete node, we have to switch
                if child.is_discrete():
                    if index < number_of_children - 1:
                        code += "tt.switch(tt.eq({name}, " \
                                "{case}), {true}, ".format(name=child.get_name(),
                                                           case=child.get_case(),
                                                           true=self._generate_code_for(node, child, case))
                    else:
                        code += self._generate_code_for(node, child, case, query)
                    switch = True
                # continuous node, we have a factor
                else:
                    if case == "mu":
                        code += "{name}*{factor}+".format(name=child.get_name(),
                                                          factor=child.get_factor()) \
                                                            + self._generate_code_for(node, child, case)
                    else:
                        assert case == "sigma", "There should be no edge from continuous to discrete nodes."
                        code += self._generate_code_for(node, child, case, query)

        # insert missing parentheses
        if switch:
            code += ")"*(number_of_children-1)

        return code
