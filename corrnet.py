
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import math
import time
import pickle
import itertools

class CorrNet(nx.Graph):
    '''
    This class is for the creation and manipulation of a correlation network to analyze
    correlated attributes individually.
    '''
    def __init__(self, df=None, info_df=None, pickle_file=None, beta=2, **kwargs):
        '''Converts raw data in df into correlation network. If present, info_df should
        be indexed by the df column name and include any relevant informtion pertaining
        to each variable. If pickle_file is present, it will read in a file that describes
        itself.
        '''
        nx.Graph.__init__(self, **kwargs)
        self.beta = beta

        if df is not None:
            # load from dataframe

            for col in df.columns:
                u = np.unique(df[col])
                if len(u) > 9:
                    vartype = 'scalar'
                else:
                    vartype = 'ordinal'
                self.add_node(col,dtype=str(df[col].dtype),vartype=vartype)

            for u in self.node:
                for v in self.node:
                    if u is not v:
                        r, p = stats.spearmanr(df[u], df[v], nan_policy='omit')
                        r, p = float(r), float(p)
                        if abs(r) > 1e-4:
                            ir = 1/(r**beta)
                        else:
                            ir = 10**16
                        self.add_edge(u,v, r=r, p=p, ir=ir,ar=abs(r))

            edge_properties = {\
                'spearman_r':'r',\
                'p-value':'p',\
                'inverse_correlation,beta={}'.format(beta):'ir',\
                'absolute_value_r':'abr'\
            }

            if info_df is not None:
                for n in self.node:
                    if n in info_df.index:
                        for col in info_df.columns:
                            self.node[n][col] = info_df.loc[n,col]
        
        elif pickle_file is not None:
            # load from pickle file

            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            self.__dict__.update(data)

    @property
    def edge_properties(self):
        return _edge_prop

    def save_pickle(self, filename):
        with open(filename, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        print('Saved CorrNet as pickle file.')

        return


def from_quantile_analysis(df, diffVar, weightFunc, centerNode, numPartitions):
    '''Splits diffVar into numPartitions quantiles for analysis.'''
    if not (self.desc.loc[diffVar,'type'] == 'rea' or self.desc.loc[diffVar,'type'] == 'ord'): # check if diffVar is a scalar value
        raise NameError(self.desc.loc[diffVar,'type'])

    percentiles = list(100/numPartitions*np.arange(numPartitions+1))        
    partCutoff = np.percentile(self.data[diffVar], percentiles)
    numPart = len(partCutoff)-1
    numNodes = len(self.corrVars)
    diffData = self.data[diffVar] # for fast comparison access

    GList = []
    i = 0
    for i in range(numPart):
        partData = self.data.loc[np.logical_and(diffData >= partCutoff[i], diffData < partCutoff[i+1]),:]
        
        gname = '%s_%1.2f_to_%1.2f_%s_cent' % (str(diffVar),partCutoff[i],partCutoff[i+1],str(centerNode))
        print('Computing partition %d resulting in graph %s.' % (i,gname))
        
        G = self.getCorrelationSpanningGraph(partData,self.desc,weightFunc,centerNode,name=gname)
        GList.append(G)

    return GList

def from_nominal_analysis(df, idf, nominal_var, verbose=False):
    '''Splits data into partitions based on nominal variable value and builds
    a separate correlation network out of each one.
    '''

    values = list(np.unique(np.array(df[nominal_var])))

    CNs = []
    i = 0
    for v in values:
        gname = '%s_%s_cent' % (str(nominal_var),str(v))

        if verbose:
            print('Computing partition %d resulting in graph %s.' % (i,gname))

        pdf = df.loc[df[nominal_var] == v]
        CN = CorrNet(pdf,idf,name=gname)
        CNs.append(CN)

        i = i + 1

    return CNs


def central_span_tree(CN, center_node, path_weight):
    '''Will create a spanning tree based on shortest paths to center_node.
    '''

    CN = CN.copy()

    # calculate shortest path between varName and every other center_node
    shortestPaths = nx.shortest_path(CN,center_node,weight=path_weight)
    shortestPathsDist = nx.shortest_path_length(CN,center_node,weight=path_weight)
    #del shortestPaths[center_node]
    #del shortestPathsDist[center_node]

    edgeInTree = {e:False for e in CN.edges()}
    distFromCenter = {}
    for n,path in shortestPaths.items():
        distFromCenter[n] = len(path)
        edgeList = [(path[i],path[i+1]) for i in range(len(path)-1)]
        for e in edgeList:
            if e in CN.edges():
                edgeInTree[e] = True
            else:
                edgeInTree[(e[1],e[0])] = True
    
    # apply node attribute indicating shortest path distance from central node
    nx.set_node_attributes(CN,'nodes_from_'+str(center_node),distFromCenter)
    nx.set_node_attributes(CN,'dist_from_'+str(center_node),shortestPathsDist)

    # create new graph where edges not in tree will be removed
    for e,inTree in edgeInTree.items():
        if not inTree:
            CN.remove_edge(*e)

    return CN

# hard threshold graph
def threshold_network(self,attrName,cutoffVal=None,cutoffNumber=None,keepLargest=True):
    '''Save edges greater than one standard deviation above the mean.'''
    # save graph that includes only the top n edges - for clarity in viewing programs
    edges = nx.get_edge_attributes(self.G,attrName)

    # decide on poorEdges depending on input
    if cutoffVal is not None:
        # take all edges above cutoffVal
        edgeNames = np.array(list(edges.keys()))
        edgeWeights = np.array(list(edges.values()))
        if keepLargest:
            poorEdges = edgeNames[edgeWeights < cutoffVal]
        else:
            poorEdges = edgeNames[edgeWeights > cutoffVal]

    elif cutoffNumber is not None:
        # keep only cutoffNumber edges (sorted is ascending by default)
        poorEdges = sorted(edges,key=edges.__getitem__,reverse=keepLargest)[0:(cutoffNumber-1)]

    # make copy of G and remove poor edges
    G = self.G.copy()
    for e in poorEdges:
        G.remove_edge(e[0],e[1])

    # also apply attributes to the special 'weight' property
    G = self.setWeightAttr(attrName,G)

    return G