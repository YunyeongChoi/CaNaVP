#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 2022

@author: yun
@purpose: Find cycle in graph. Used for identify all ternary zone in Ca-Na-VP phase diagram.
This used for calculation chemical potential of Ca, Na and VPO.

# Need to be moved to somewhere else. Need polishing.
"""
import numpy as np
from compmatscipy.CompAnalyzer import CompAnalyzer


class ternary_chempo:

    def __init__(self, edges, data):
        """
        Args:
            edges: list of tuples with form [(std formula, std formula)]
            data: energy dictionary with form {'std formula': energy per atom}
        """
        self.edges = edges
        self.data = data
        self.cycles = []
        self.vertices = []
        for edge in self.edges:
            for vertex in edge:
                if vertex not in self.vertices:
                    self.vertices.append(vertex)
        self.parse_cycles()

    @staticmethod
    def gcd(a, b, rtol=1e-03, atol=1e-03):
        # Need caution. Do not use after rounding floats < 3.
        t = min(abs(a), abs(b))
        while abs(b) > rtol * t + atol:
            a, b = b, a % b
        return a

    @staticmethod
    def invert(path) -> list:

        return ternary_chempo.rotate_to_smallest(path[::-1])

    @staticmethod
    #  rotate cycle path such that it begins with the smallest node
    def rotate_to_smallest(path) -> list:

        n = path.index(min(path))
        return path[n:] + path[:n]

    @staticmethod
    def is_visited_node(node, path) -> bool:

        return node in path

    @staticmethod
    def cmpd_to_fraction(cmpd) -> tuple:
        """
        Args:
            Standard Formula
        Returns:
            tuple of Ca, Na in V2(PO4)3
        """
        ca = CompAnalyzer(cmpd)
        reduced_ratio = (ca.amt_of_el('O')) / 12
        ca_amt = ca.amt_of_el('Ca') / reduced_ratio
        na_amt = ca.amt_of_el('Na') / reduced_ratio

        return ca_amt, na_amt

    @staticmethod
    def fraction_to_cmpd(fraction) -> str:
        """
        Args:
            tuple of Ca, Na in V2(PO4)3
        Returns:
            Standard Formula
        """
        ratio = ternary_chempo.gcd(fraction[0], fraction[1])
        for i in [2, 3, 12]:
            ratio = ternary_chempo.gcd(ratio, i)

        formula = ''
        if not fraction[1] == 0:
            formula += 'Na' + str(fraction[1] / ratio)
        if not fraction[0] == 0:
            formula += 'Ca' + str(fraction[0] / ratio)

        formula += 'V' + str(np.round(2 / ratio, 3)) + \
                   'P' + str(np.round(3 / ratio, 3)) + \
                   'O' + str(np.round(12 / ratio, 3))
        ca = CompAnalyzer(formula)

        return ca.std_formula()

    @staticmethod
    def is_three_on_line(x, y, z) -> bool:

        return (x[0] * (y[1] - z[1]) + y[0] * (z[1] - x[1]) + z[0] * (x[1] - y[1])) == 0

    @staticmethod
    def is_point_in_triangle(s, x, y, z) -> bool:

        sx = [s[0] - x[0], s[1] - x[1]]
        s_xy = ((y[0] - x[0]) * sx[1] - (y[1] - x[1]) * sx[0]) > 0

        if ((z[0] - x[0]) * sx[1] - (z[1] - x[1]) * sx[0] > 0) == s_xy:
            return False
        if ((z[0] - y[0]) * (s[1] - y[1]) - (z[1] - y[1]) * (s[0] - y[0]) > 0) != s_xy:
            return False
        if s == x or s == y or s == z:
            return False

        return True

    def is_new_path(self, path) -> bool:

        return not (path in self.cycles)

    def find_new_cycles(self, path) -> None:
        """
        Args:
            path: line connecting vertices.
        Returns:
            Add new cycle to self.cycles.
            This does not consider line or non-smallest cycles.
        """
        start_node = path[0]

        # visit each edge and each node of each edge
        for edge in self.edges:
            node1, node2 = edge
            if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
                if not ternary_chempo.is_visited_node(next_node, path):
                    # neighbor node not on path yet
                    sub = [next_node]
                    sub.extend(path)
                    # explore extended path
                    self.find_new_cycles(sub)
                elif len(path) == 3 and next_node == path[-1]:
                    # cycle found
                    p = ternary_chempo.rotate_to_smallest(path)
                    inv = ternary_chempo.invert(p)
                    if self.is_new_path(p) and self.is_new_path(inv):
                        self.cycles.append(p)

    def parse_cycles(self) -> None:
        """
        Returns:
            None. update self.cycles.
            parse cycles that are in same line or non-smallest.
        """
        for edge in self.edges:
            for node in edge:
                self.find_new_cycles([node])

        # change string to fraction of Ca, Na.
        fractions = []
        for i in self.cycles:
            temp = []
            for j in i:
                temp.append(ternary_chempo.cmpd_to_fraction(j))
            fractions.append(temp)

        # For removing line or triangle that has point in it.
        min_cycle_list = []
        for i in fractions:
            if not ternary_chempo.is_three_on_line(i[0], i[1], i[2]):
                point_in = False
                for j in self.vertices:
                    if ternary_chempo.is_point_in_triangle(ternary_chempo.cmpd_to_fraction(j),
                                                           i[0], i[1], i[2]):
                        point_in = True
                        break
                if point_in:
                    continue
                else:
                    min_cycle_list.append(i)

        self.cycles = []
        for i in min_cycle_list:
            temp = []
            for j in i:
                temp.append(ternary_chempo.fraction_to_cmpd(j))
            self.cycles.append(tuple(temp))

        return

    def get_chempo_at_one_cycle(self, x, y, z) -> tuple:
        """
        Args:
            x, y, z: str of three coordinates in cycles. In std formula.
        Returns:
            Chemical potential of Ca/Na/VPO
        """
        concentrations = np.zeros((3, 3))
        energies = np.zeros((3, 1))
        for i, j in enumerate([x, y, z]):
            ca, na = ternary_chempo.cmpd_to_fraction(j)
            concentrations[i, :] = np.array([ca, na, 17])
            energies[i] = self.data[j] * (ca + na + 17)

        chempo = np.linalg.inv(concentrations) @ energies

        return chempo

    """
        Ca_voltage = (-chempo[0][0] - 2.0056) / 2
        Na_voltage = (-chempo[1][0] - 1.3225) / 1
    """

    def get_chempo_at_cycles(self):

        chempo_dict = {}
        for i in self.cycles:
            chempo_dict[i] = self.get_chempo_at_one_cycle(i[0], i[1], i[2])

        return chempo_dict
