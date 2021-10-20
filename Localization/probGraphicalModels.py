# probGraphicalModels.py - Graphical Models and Belief Networks
# AIFCA Python3 code Version 0.9.1 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from probFactors import CPD
from Main.display import Displayable
import matplotlib.pyplot as plt


class GraphicalModel(Displayable):
    """The class of graphical models. 
    A graphical model consists of a title, a set of variables and a set of factors.

    vars is a set of variables
    factors is a set of factors
    """
    def __init__(self, title, variables=None, factors=None):
        self.title = title
        self.variables = variables
        self.factors = factors

class BeliefNetwork(GraphicalModel):
    """The class of belief networks."""

    def __init__(self, title, variables, factors):
        """vars is a set of variables
        factors is a set of factors. All of the factors are instances of CPD (e.g., Prob).
        """
        GraphicalModel.__init__(self, title, variables, factors)
        assert all(isinstance(f,CPD) for f in factors)
        self.var2cpt = {f.child:f for f in factors}
        self.var2parents = {f.child:f.parents for f in factors}
        self.children = {n:[] for n in self.variables}
        for v in self.var2parents:
            for par in self.var2parents[v]:
                self.children[par].append(v)
        self.topological_sort_saved = None

    def topological_sort(self):
        """creates a topological ordering of variables such that the parents of
        a node are before the node.
        """
        if self.topological_sort_saved:
            return self.topological_sort_saved
        next_vars = {n for n in self.var2parents if not self.var2parents[n] }
        self.display(3,'topological_sort: next_vars',next_vars)
        top_order=[]
        while next_vars:
            var = next_vars.pop()
            self.display(3,'select variable',var)
            top_order.append(var)
            next_vars |= {ch for ch in self.children[var]
                              if all(p in top_order for p in self.var2parents[ch])}
            self.display(3,'var_with_no_parents_left',next_vars)
        self.display(3,"top_order",top_order)
        assert set(top_order)==set(self.var2parents),(top_order,self.var2parents)
        self.topologicalsort_saved=top_order
        return top_order

    def show(self):
        plt.ion()   # interactive
        ax = plt.figure().gca()
        ax.set_axis_off()
        plt.title(self.title)
        bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5")
        for var in reversed(self.topological_sort()):
            if self.var2parents[var]:
                for par in self.var2parents[var]:
                    ax.annotate(var.name, par.position, xytext=var.position,
                                    arrowprops={'arrowstyle':'<-'},bbox=bbox,
                                    ha='center')
            else:
                x,y = var.position
                plt.text(x,y,var.name,bbox=bbox,ha='center')


class InferenceMethod(Displayable):
    """The abstract class of graphical model inference methods"""
    method_name = "unnamed"  # each method should have a method name

    def __init__(self,gm=None):
        self.gm = gm

    def query(self, qvar, obs={}):
        """returns a {value:prob} dictionary for the query variable"""
        raise NotImplementedError("InferenceMethod query")   # abstract method


    
