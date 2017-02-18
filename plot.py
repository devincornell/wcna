import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def edge_hist(CN, attr, bins=10, cutoff=None):
    edges = np.array(list(nx.get_edge_attributes(CN,attr).values()))
    
    if cutoff is not None:
        u = np.median(edges)
        std = np.sqrt(np.mean(np.absolute(edges-u)))*cutoff
        edges = edges[np.logical_and(edges > u-std, edges < u+std)]

    #hist,binedges = np.histogram(edges,bins=len(edges)*bins)
    nbins = math.floor(bins*math.log10(np.size(edges)))
    plt.hist(edges,nbins)
    plt.show()

def node_hist(CN, attr, bins=10):
    edges = nx.get_node_attributes(CN,attr)

    weightList = []
    for eKey in edges.keys():
        weightList.append(edges[eKey])

    weightList = np.array(weightList)
    hist,binedges = np.histogram(weightList,bins=len(edges)*bins)

    plt.plot(binedges[0:-1],hist)
    plt.show()