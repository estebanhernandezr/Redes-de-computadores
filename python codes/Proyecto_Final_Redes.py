# Redes de Computadores 2022
# Universidad del Rosario - School of Engineering, Science and Technology
# Pr. David Celeita
import numpy as np
import pandas as pd
import copy
import scipy as sp
import networkx as nx
from scipy import sparse
import time
import random as rd
import csv
import matplotlib.pyplot as plt

class Node:
    # Class that defines a node of a graph
    def __init__(self, id):
        # Class constructor
        # INPUT:
        #   id: int that identifies node
        self.id = id
        self.neighs = []
        self.distance = np.inf
        self.predecessor = None

    def add_neigh(self, node, weight):
        # Adds nodes to the neighs list of the node
        # INPUT:
        #   node: object Node
        #   weight: weight of the edge
        self.neighs.append((node, float(weight)))

class Graph(Node):
    # Class that creates a graph
    def __init__(self, sparse_mat):
        # Class constructor
        # INPUT:
        #   sparse_mat: np.array with [node A, node B, weight] as rows

        self.nodes = []
        self.sparse = sparse_mat

        for row in sparse_mat:
            node_A = row[0]
            node_B = row[1]
            weight = row[2]

            A_in_graph = self.node_in_graph(node_A)
            B_in_graph = self.node_in_graph(node_B)

            if A_in_graph == False:
                self.nodes.append(Node(node_A))
            
                if B_in_graph == False:
                    self.nodes.append(Node(node_B))
                    self.nodes[-2].add_neigh(self.nodes[-1], weight)
                    self.nodes[-1].add_neigh(self.nodes[-2], weight)

                else: # B_in_graph -> index where node B is in self.nodes
                    self.nodes[-1].add_neigh(self.nodes[B_in_graph-1], weight)
                    self.nodes[B_in_graph-1].add_neigh(self.nodes[-1], weight)
                
            else:
                if B_in_graph == False:
                    self.nodes.append(Node(node_B))
                    self.nodes[-1].add_neigh(self.nodes[A_in_graph-1], weight)
                    self.nodes[A_in_graph-1].add_neigh(self.nodes[-1], weight)

                else:
                    self.nodes[A_in_graph-1].add_neigh(self.nodes[B_in_graph-1], weight)
                    self.nodes[B_in_graph-1].add_neigh(self.nodes[A_in_graph-1], weight)

    def node_in_graph(self, node_id):
        # determines whether or not a node is in a graph. Looks for it in self.nodes
        # INPUT:
        #   node_id: int that specifies the node to look for
        # OUTPUT:
        #   returns index of the node in the list if exists, if not return False
        for idx, node in enumerate(self.nodes):
            if node.id == node_id: 
                return idx+1
        return False

    def _min_dist(self, nodes):
        # determines the minimum distance of a list of nodes
        # INPUT:
        #   nodes: list of nodes to look on
        # OUTPUT:
        #   min_node: the node with the minimum distance
        min_node = nodes[0]
        min_dist = min_node.distance
        for i in range(1, len(nodes)):
            if nodes[i].distance < min_dist:
                min_dist = nodes[i].distance
                min_node = nodes[i]
        return min_node

    def sparse2adj(self, sparse):
        # transforms a sparse matrix into a adjacency matrix
        # INPUT:
        #   sparse: numpy array - sparse matrix of graph
        # OUTPUT:
        #   adj: numpy array - adjacency matrix of graph
        adj = np.zeros((len(self.nodes),len(self.nodes)))
        for row in sparse:
            node_A = int(row[0])
            node_B = int(row[1])
            weight = float(row[2])
            adj[node_A][node_B] = weight
            adj[node_B][node_A] = weight
        return adj

    def draw_graph(self):
        # draws each node and its corresponding edges
        G = nx.convert_matrix.from_numpy_array(self.sparse2adj(self.sparse))
        pos = nx.spring_layout(G)
        labels = {}
        for i in self.sparse:
            labels.setdefault((i[0],i[1]),i[2])
        plt.title("Graph visualization")
        nx.draw(G, pos, with_labels = True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
        plt.show()
        
    def routing_table(self):
        # determines and prints the routing table the graph
        # OUTPUT:
        #   table: DataFrame of the routing table of graph
        id = []
        dist = []
        pred = []
        for node in self.nodes:
            id.append(node.id)
            dist.append(node.distance)
            pred.append(node.predecessor)
        table = np.column_stack((id,dist,pred))
        table = pd.DataFrame(table)
        table.columns = ["id", "distance", "predecessor"]
        print("\n ----------------- Routing Table ---------------------- \n")
        print(table.to_string(index=False))
        return table

    def draw_tree(self, route_table):
        # draws the tree corresponding to the routing table
        # INPUT:
        #   routing_table: DataFrame with the routing table
        route_m = route_table.to_numpy()
        route_m[:, [2, 1]] = route_m[:, [1, 2]] 
        route_m = np.delete(route_m, (0), axis=0)
        G = nx.convert_matrix.from_numpy_array(self.sparse2adj(route_m))
        plt.title("Final Tree")
        nx.draw(G, with_labels = True)
        plt.show()
    
    def dijkstra(self):
        # Performs Dijkstra's algorithm and plots the original graph and the resulting tree

        # starting time
        start = time.time()

        actual = self.nodes[0]
        actual.distance = 0
        actual.predecessor = -1
        unvisited = copy.copy(self.nodes)

        it = 0
        print("\n -------------- Execution Time per Iteration ----------- \n")

        while len(unvisited) > 0:
            it += 1
            # iteration start time
            start_i = time.time()
            for neigh in actual.neighs:
                if neigh[0] in unvisited: #se puede crear un atributo
                    temp_dist = actual.distance + neigh[1]
                    if temp_dist < neigh[0].distance:
                        neigh[0].distance = temp_dist
                        neigh[0].predecessor = actual.id
            unvisited.remove(actual)
            if len(unvisited) != 0:
                actual = self._min_dist(unvisited)
            # iteration end time
            end_i = time.time()
            print("Runtime of iteration {0} in Dijkstra's Algorithm is {1:.50f}".format(it, end_i - start_i))

        # end time
        end = time.time()
        # total time taken
        print("\n ----------------- Execution Time ---------------------- \n")

        print(f"Runtime of Dijkstra's Algorithm is {end - start}")

        print("\n ----------------- Number of Iterations ---------------- \n")
        print("Dijkstra took {0} iterations to finish".format(it))

        self.draw_tree(self.routing_table())


def create_random_sparse(nodos = 15):
    # Creates a random sparse matrix given a number of nodes
    # INPUT:
    #   nodos: The number of nodes in the graph
    # OUTPUT:
    #   mA: numpy array of the sparse matriz
    nodos = int(nodos)
    i = 0
    while i <= nodos:
        tmp = np.array([np.random.randint([0, i, 1], [nodos, i+1, 15])])
        if tmp[0][0] != tmp[0][1]:
            if i == 0:
                mA = np.array([np.random.randint([0, i, 1], [nodos, i+1, 15])])
            else:
                mA = np.concatenate((mA, tmp), axis=0)
            i+=1
    i = 0
    while i<int(nodos/2):
        tmp = np.array([np.random.randint([0, 0, 1], [nodos, nodos, 15])])
        if tmp[0][0] != tmp[0][1]:
            mA = np.concatenate((mA, tmp), axis=0)
            i+=1

    return mA

def import_csv(filename):
    # Reads a csv and converts to numpy array
    # INPUT:
    #   filename: name of the csv file
    # OUTPUT:
    #   numpy array of the csv file
    df = pd.read_csv(filename, sep=';')
    return df.to_numpy()

def import_txt(filename):
    # Reads a txt and converts to numpy array
    # INPUT:
    #   filename: name of the txt file
    # OUTPUT:
    #   numpy array of the txt file
    f = open(filename, "r")
    first1 = True
    text = f.readline()
    while text != '':
        a = ''
        b = ''
        c = ''
        start = 'a'
        for i in text:
            if i != ' ' and start == 'a':
                a+=i
            elif i!= ' ' and start == 'b':
                b+=i
            elif i != '\n' and start == 'c':
                c+=i
            else:
                if start == 'a':
                    start = 'b'
                elif start == 'b':
                    start = 'c'
        if first1:
            mT = np.array([[int(a),int(b),int(c)]])
            first1 = False
        else:
            tmp = np.array([[int(a),int(b),int(c)]])
            mT = np.concatenate((mT, tmp), axis=0)
        text = f.readline()
    return mT
    
