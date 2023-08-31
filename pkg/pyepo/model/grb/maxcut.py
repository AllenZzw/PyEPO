#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class maxCutModel(optGrbModel):
    """
    This class is optimization model for maximum independent set problem 

    Attributes:
        _model (GurobiPy model): Gurobi model
        graph (networkx graph): Graph structure 
        arcs (list): List of arcs
    """

    def __init__(self, graph):
        """
        Args:
            graph (networkx graph): size of grid network
        """
        self.graph = graph
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("maxcut")
        # varibles
        x = m.addVars(self.graph.edges, name="x", vtype=GRB.BINARY)
        y = m.addVars(self.graph.nodes, name="y", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for e in self.graph.edges:
            m.addConstr(x[e] <= y[e[0]] + y[e[1]])
            m.addConstr(x[e] <= 2 - (y[e[0]] + y[e[1]]))

        return m, x

        