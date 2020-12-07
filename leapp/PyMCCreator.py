import itertools
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse


class PyMCCreator(object):
    def __init__(self, bayesian_model):
        self.bm = bayesian_model
        self.level_dict = self.bm.get_level_dict()

    def generate(self, function_name="trace", with_model=True, environments=True, as_function=False, with_trace=False, number_of_samples=1000, query=None):
        code = ""
        level= ""
        if not with_model:
            as_function = False
            with_trace = False
        if with_model:
            if environments:
                code = "import pymc3 as pm\n"
                code += "import theano.tensor as tt\n"
            if query:
                if environments:
                    code += "import numpy as np\n"
                    code += "from sklearn.metrics import accuracy_score, mean_absolute_error\n"
                    # we need this method for the mean_absolute_error
                    code += "def mep(samples, bins=1000):\n"
                    level = "    "
                    code += f"{level}# get bins and boarders\n"
                    code += f"{level}b, m = np.histogram(samples, bins=bins)\n"
                    code += f"{level}# get the boarders for the bin\n"
                    code += f"{level}d, c = (m[[k for k, (i, j) in enumerate(zip(b, m)) if i == max(b)][0] - 1], m[[k for k, (i, j) in enumerate(zip(b, m)) if i == max(b)][0]])\n"
                    code += f"{level}# get the samples inside of this bin\n"
                    code += f"{level}s1 = sorted(samples)\n"
                    code += f"{level}s2 = samples[samples >= d]\n"
                    code += f"{level}s3 = s2[s2 <= c]\n"
                    code += f"{level}# return the mean of this bin\n"
                    code += f"{level}return np.mean(s3)\n"
                    level = ""
            # len of all data, you have to set the variable data by yourself later
            if query:
                if environments:
                    code += f"{level}data = None\n"
                    code += f"{level}n = len(data.values)\n"
                    # parameter for sampling
                    code += f"{level}tune = 1000\n"
                    code += f"{level}number_of_samples=10000\n"
                    # predicted values
                    code += f"{level}y_pred = []\n"
            if as_function:
                code += f"def {function_name}():\n"
                level += "    "
            if query:
                code += f"{level}print({function_name})\n"
                code += f"{level}logging.info('start with {function_name}')\n"
                code += f"{level}y_pred = []\n"
                code += f"{level}for index, data_point in data.iterrows():\n"
                level += "    "
                code += f"{level}print('current data point', index)\n"
            code += f'{level}with pm.Model() as model:\n'
            level += "    "
        insertion_order = self.bm.get_condition_node_order()

        for node in insertion_order:
            # from bayesian_network_learning.util.ProbParameter import print_prob_table; print_prob_table(node.get_parameter().get_graph(), node.get_name(), True);
            name = node.get_name()
            code += f"{level}{name} = "
            tree = node.get_parameter().get_prob_graph().get_head()

            if node.is_discrete():
                code += f"pm.Categorical('{name}', "
                # if we have a query we look if we have to add the observed parameter
                if query != None and name in query['conditions']:
                    code += "observed=data_point['{name}'], "
                code += "p={prob})\n".format(name=name, prob=self._generate_code_for(node, tree, "prob", query))
            else:
                code += f"pm.Normal('{name}', "
                # if we have a query we look if we have to add the observed parameter
                if query != None and name in query['conditions']:
                    code += f"observed=data_point['{name}'], "
                code += "mu={mu}, " \
                        "sigma={sigma})\n".format(name=name,
                                                  mu=self._generate_code_for(node, tree, "mu", query),
                                                  sigma=self._generate_code_for(node, tree, "sigma", query))
        if with_trace:
            code += f"{level}return pm.sample({number_of_samples})\n"
        if query != None:
            # one level back
            code += f"{level[:-4]}with model:\n"
            code += f"{level}samples = pm.sample(draws=number_of_samples, tune=tune, model=model)\n"
            # one level back and count the appearances
            if query['score'] == 'accuracy':
                code += f"{level[:-4]}y_pred.append([i for i, j in zip(np.unique(samples['{query['variable']}']), np.bincount(samples['{query['variable']}']))][0])\n"
                # two levels back
                code += f"{level[:-8]}score = accuracy_score(y_pred=y_pred, y_true=data['{query['variable']}'])\n"
                code += f"{level[:-8]}print(score)\n"
            elif query['score'] == 'mean_absolute':
                code += f"{level[:-4]}y_pred.append(mep(samples['{query['variable']}']))\n"
                # two levels back
                code += f"{level[:-8]}score = mean_absolute_error(y_pred=y_pred, y_true=data['{query['variable']}'])\n"
                code += f"{level[:-8]}print(score)\n"
                code += f"{level[:-8]}logging.info(f'{function_name}: {{score}}')\n"
                code += f"{level[:-8]}return score\n"
        return code

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
