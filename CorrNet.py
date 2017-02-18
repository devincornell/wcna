
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import math
import time
import pickle
import itertools
#from numba import jit

#import mcl # markov chain library

class CorrNet(nx.Graph):
    '''
    This class is for the creation and manipulation of a correlation network to analyze
    correlated attributes individually.
    '''
    def __init__(self, df=None, pickle_file=None, name=None, beta=2, *args):
        '''Converts raw data into a Correlation Network'''
        nx.Graph.__init__(self, *args)

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
                        self.add_edge(u,v, r=r, p=p, ir=1/(r**beta),ar=abs(r))
        
        elif pickle_file is not None:
            # load from pickle file

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.__dict__.update(data)

        if name is not None:
            self.graph['name'] = name

    ##### Get Node or Edge Summary Statistics #####
    '''
        These are graph functions that allow the user to see summary statistics about
        weights that were created.
    '''

    def plotEdgeWeightHist(self, attrName,relBins=10,outlierCutoff=None):
        edges = np.array(list(nx.get_edge_attributes(self.G,attrName).values()))
        
        if outlierCutoff is not None:
            u = np.median(edges)
            std = np.sqrt(np.mean(np.absolute(edges-u)))*outlierCutoff
            edges = edges[np.logical_and(edges > u-std, edges < u+std)]

        #hist,binedges = np.histogram(edges,bins=len(edges)*relBins)
        nbins = math.floor(relBins*math.log10(np.size(edges)))
        plt.hist(edges,nbins)
        plt.show()

    def plotNodeWeightHist(self, attrName,relBins=1.0):
        edges = nx.get_node_attributes(bn.G,attrName)

        weightList = []
        for eKey in edges.keys():
            weightList.append(edges[eKey])

        weightList = np.array(weightList)
        hist,binedges = np.histogram(weightList,bins=len(edges)*relBins)

        plt.plot(binedges[0:-1],hist)
        plt.show()

    def getRankedNodes(self,attrName,descending=False):
        attr = nx.get_node_attributes(self.G,attrName)
        ordered = sorted(tuple(attr),key=attr.__getitem__,reverse=descending)
        orderedList = [(x, attr[x]) for x in ordered]
        return ordered

    def getRankedEdges(self,attrName,descending=False):
        attr = nx.get_edge_attributes(self.G,attrName)
        ordered = sorted(tuple(attr),key=attr.__getitem__,reverse=descending)
        orderedList = [(x, attr[x]) for x in ordered]
        return ordered


    ##### Apply Graph Edge Weights #####
    '''
        These are graph functions that can be added as attributes to the main graph.
    '''

    def applyWeightAttr(self, attrName, attrFunc, baseAttr='weight'):
        '''Create a new weight attribute described by attrFunc(baseAttr).'''

        # apply icorr attribute
        edges = self.G.edges(data=True)
        icorrWeights = {}
        for e in edges:
            icorrWeights[(e[0],e[1])] = float(attrFunc(e[2][baseAttr]))
        nx.set_edge_attributes(self.G,attrName,icorrWeights)

        return


    
    def applyMCLClusterAttr(self, attrName, baseAttr, numClusters):

        # run mcl clustering algorithm
        mcl.cluster_mcl(self.G,numClusters,verbose=True,weightAttr=baseAttr)

        if clusters is not None:
            nx.set_node_attributes(self.G,'cluster',clusters)
        else:
            # assign all nodes to be in cluster zero
            nodes = G.nodes()
            clusters = {}
            for n in nodes:
                clusters[n] = 0
            nx.set_node_attributes(self.G,attrName,clusters)

        return

    ##### Create Derived Graphs From Original CorrNet ##### 
    '''
        These are graph functions that result in derived graphs for analysis.
    '''

    # hard threshold graph
    def getHardThresholdGraph(self,attrName,cutoffVal=None,cutoffNumber=None,keepLargest=True):
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

    def getMinSpanningTree(self,weightAttr):
        G = nx.minimum_spanning_tree(self.G.copy(),weightAttr)
        G = self.setWeightAttr(weightAttr,G)
        return G

    def getCentralizedSpanningTree(self,nodeName,attrName, Gin = None):
        '''Will create a spanning tree based on shortest paths to nodeName.'''

        # calculate shortest path between varName and every other nodeName
        shortestPaths = nx.shortest_path(self.G,nodeName,weight=attrName)
        shortestPathsDist = nx.shortest_path_length(self.G,nodeName,weight=attrName)
        del shortestPaths[nodeName]
        del shortestPathsDist[nodeName]

        # mark which, of all existing edges, are in these shortest paths
        if Gin is None:
            G = self.G.copy()
        else:
            G = Gin.copy()

        edges = G.edges(data=False)
        edgeInTree = {e:False for e in edges}
        distFromCenter = {}
        for n,path in shortestPaths.items():
            distFromCenter[n] = len(path)
            edgeList = [(path[i],path[i+1]) for i in range(len(path)-1)]
            for e in edgeList:
                if e in edges:
                    edgeInTree[e] = True
                else:
                    edgeInTree[(e[1],e[0])] = True
        
        # apply node attribute indicating shortest path distance from central node
        nx.set_node_attributes(G,'nodes_from_'+str(nodeName),distFromCenter)
        nx.set_node_attributes(G,'dist_from_'+str(nodeName),shortestPathsDist)

        # create new graph where edges not in tree will be removed
        for e,inTree in edgeInTree.items():
            if not inTree:
                G.remove_edge(*e)

        return G

    def getNominalAnalysisGraphs(self, diffVar, weightFunc, centerNode):
        '''Splits data into partitions based on nominal variable value.'''
        if not self.desc.loc[diffVar,'type'] == 'nom': # check if diffVar is a scalar value
            raise NameError(diffVar)

        partVals = list(np.unique(np.array(self.data[diffVar])))
        numPart = len(partVals)
        numNodes = len(self.corrVars)

        #corrMat = np.zeros((numNodes,numNodes,numPart))
        GList = []
        i = 0
        for v in partVals:
            partData = self.data.loc[self.data[diffVar] == v]
            
            gname = '%s_%s_%s_cent' % (str(diffVar),str(v),str(centerNode))
            print('Computing partition %d resulting in graph %s.' % (i,gname))

            G = self.getCorrelationSpanningGraph(partData,self.desc,weightFunc,centerNode,name=gname)
            GList.append(G)

            i = i + 1

        return GList

    #def getCorrelationSpanningGraph(self,data,desc,graphAttr,weightFunc,centerNode,name='')
    def getQuantileAnalysisGraphs(self, diffVar, weightFunc, centerNode, numPartitions):
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

    def getCorrelationSpanningGraph(self,partData,desc,weightFunc,centerNode,name=''):
        cn = CorrNet(partData,self.desc, 'correlation',netName=name)
            
        # calculate weight function and find central node tree
        cn.applyWeightAttr('centered_tree_weight',weightFunc,baseAttr='correlation')
        G = cn.getCentralizedSpanningTree(centerNode,'centered_tree_weight')

        G.graph['name'] = name # assign this for later use

        return G
    
    def save_pickle(self, filename):
        with open(filename, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        print('Saved CorrNet as pickle file.')

        return




