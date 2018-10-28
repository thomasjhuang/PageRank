import os
import sys
import math
import json
import numpy as np
from collections import OrderedDict
from numpy import linalg as LA
from scipy.sparse import csc_matrix
import pandas

def printRanks(pr, url_dict):
    id_to_url = dict((v,k) for k,v in url_dict.items())
    url_to_pr = {}
    pr_to_url = {}

    for i in range(len(pr)):
        url_to_pr[str(id_to_url[str(i)])]  = pr[i]
        pr_to_url[pr[i]] = str(id_to_url[str(i)])

    i = 0
    for k,v in sorted(pr_to_url.items(), reverse=True):
        if(i > 20):
            break
        print(v + "\t" + str(k))
        i += 1

def create_nodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes

def read_in(adj_list, url_dict):
    id_to_index = {}
    transition_mat = []

    for i in range(len(url_dict)):
        transition_mat.append([0] * len(url_dict))
        if str(i) in adj_list:
            links = adj_list[str(i)]
            for link in links:
                transition_mat[i][int(link)] = float(1.0)
        else:
            transition_mat[i] = list(map(lambda x: 1 / len(transition_mat[i]), transition_mat[i]))
            
    array = np.array(transition_mat)
    return array, url_dict

def pageRank(transition_matrix, max_iterations, epsilon):

    transition_matrix = pandas.DataFrame(transition_matrix)
    #creating initial graph
    nodes = set()
    for colKey in transition_matrix:
        nodes.add(colKey)
    for rowKey in transition_matrix.T:
        nodes.add(rowKey)

    #makes sure values are positive
    matrix = transition_matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(np.ones(len(matrix[colKey])), index=matrix.index)
    transition_matrix = matrix.T

    #normalization
    normalize = 1.0 / float(len(nodes))
    P_matrix =  pandas.Series({node : normalize for node in nodes})

    #normalizes the rows
    transitionProbs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    #iterate until convergence
    for iteration in range(max_iterations):
        old = P_matrix.copy()
        P_matrix = P_matrix.dot(transitionProbs)
        delta = P_matrix - old
        if math.sqrt(delta.dot(delta)) < epsilon:
            break

    return P_matrix

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('--max_iterations', help='Total amount of iterations',
                  type=float, default=0.15)

    arg_parser.add_argument('--epsilon', help='convergence detection parameter',
                  type=float, default=0.05)

    arg_parser.add_argument('--adj_list', help='path to the adjacency list file (JSON)',
                 type=str, default='./adj_list.json')

    arg_parser.add_argument('--url_dict', help='path to URL to ID mapping file (JSON)',
                 type=str, default='./url_dict.json')

    args = arg_parser.parse_args()
    adj_list = json.load(open('./adj_list.json'))
    url_dict = json.load(open('./url_dict.json'))
    array, url_dict = read_in(adj_list, url_dict)
    pr = pageRank(array)
    printRanks(pr, url_dict)
