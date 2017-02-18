
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
import pickle
import itertools
#from numba import jit

import mcl # markov chain library

class CorrNet(nx.Graph):
    '''
    This class is for the creation and manipulation of a correlation network to analyze
    correlated attributes individually.

    '''
    def __init__(self,data=None,descriptions=None,corrAttrName=None,pickleFile=None,netName=None):
        '''Converts raw data into a Correlation Network'''
        nx.Graph.__init__(self)
        
        if data is not None and descriptions is not None and corrAttrName is not None:
            self.constructCorrWeightGraph(data,descriptions,corrAttrName)        
        
        elif pickleFile is not None:
            self.loadFromPickle(pickleFile)

        if netName is not None:
            self.name = netName
        else:
            self.name = ''


    def constructCorrWeightGraph(self, data, desc, corAttrName):
        '''Takes data and description tables (both Pandas df) and converts them into a graph where weights are correlations.'''
        
        # validvar list is intersection of data and desc
        datavars = data.columns.values.tolist() # list of variables
        descvars = desc.index.values.tolist()
        validVars = list(set(datavars) & set(descvars))
        
        # separate into correlation vars and nominal vars
        corrVars = []
        nomVars = []
        for v in validVars:
            if desc.loc[v,'type'] == 'nom':
                nomVars.append(v)
            else:
                corrVars.append(v)

        # extract needed portions of data and descriptions
        data = data.loc[:,corrVars + nomVars]
        desc = desc.loc[corrVars + nomVars,:]


        # get correlation matrix and construct graph
        corrMat = self.getCorrWeightMat(data.loc[:,corrVars],desc,corrVars)
        
        # create graph using node names and correlation matrix
        G = graphFromMat(corrMat, corrVars, corAttrName)

        # apply attributes from desc about variables
        descAttr = desc.loc[corrVars,:].to_dict().items()
        for key, val in descAttr:
            nx.set_node_attributes(G,key,val)

        # assign as member vars to CorrNet
        self.G = G.copy()
        self.corrVars = corrVars
        self.nomVars = nomVars
        self.data = data
        self.desc = desc

        return

    def getCorrWeightMat(self, data, desc, corrVars):

        # perform fast correlations on all variables
        corrMat = data.loc[:,corrVars].corr(method='pearson') # for scalars (write over this one)
        spearmanCorr = data.loc[:,corrVars].corr(method='spearman') # for ordinals

        # apply spearman correlation where needed
        for v in corrVars:
            if desc.loc[v,'type'] == 'ord':
                # set row & col as spearman correlation
                corrMat.loc[v,:] = spearmanCorr.loc[v,:]
                corrMat.loc[:,v] = spearmanCorr.loc[:,v]

        return np.array(corrMat)


    ##### Get Properties of Graph #####
    '''
        These functions allow basic access and adjustment of graph
        properties.
    '''
    def getEdgeProperties(self):
        edgeDat = self.G.get_edge_data(self.var[0],self.var[1])
        return edgeDat.keys()

    def getNodeProperties(self):
        return None

    def getVarNames(self):
        return self.var


    def setWeightAttr(self,attrName,G=None):
        '''Set attrName values as the special weight attribute.
            If a graph is provided as input, it will output a 
            new graph with the weight attribute set.'''
        if G is not None:
            attrG = G.copy()
            weightAttr = nx.get_edge_attributes(attrG,attrName)
            nx.set_edge_attributes(attrG,'weight',float(weightAttr))
            return attrG
        else:
            weightAttr = nx.get_edge_attributes(self.G,attrName)
            nx.set_edge_attributes(self.G,'weight',float(weightAttr))
            return

        # set attribute as 'weight'

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

    def applyCentralityAttr(self, attrName, baseAttr):
        # add attributes for centrality to each node
        bc = nx.betweenness_centrality(self.G,weight=baseAttr)
        nx.set_node_attributes(self.G,attrName,bc)
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

    ##### Graph Clustering Algorithms #####

    def analyzeMCLClusters(self,weightAttr,rRange=np.arange(2,30),filename=None):
        mcl.analyze_mcl(self.G,rRange=rRange,showPlot=True,weightAttr=weightAttr,filename=filename)
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

    def getCorrelationGraph(self):
        return self.G.copy()

    def getDiffGraph(self, diffAttr, otherNet):
        '''This function will get a graph representing differences between the CorrNets of
            two different differentials of a varable's distribution. It assumes that both
            graphs have the same variables and same sets of edges.'''
        
        # new graph describing differential
        G = nx.Graph()
        G.add_nodes_from(self.G.nodes())

        # get edges from self and set them in new graph
        edges = nx.get_edge_attributes(self.G,diffAttr)
        nx.set_edge_attributes(G, self.name+'_'+diffAttr, edges)
        
        # get edges from self and set them in new graph
        otherEdges = nx.get_edge_attributes(otherNet.G,diffAttr)
        nx.set_edge_attributes(G,otherNet.name+'_'+diffAttr,otherEdges)

        eDiff = {} # populate this with edge differentials
        for e in edges:
            eDiff[e] = math.fabs(edges[e] - otherEdges[e])

        # apply eDiff as weight attribute and as a 'diff' attribute
        nx.set_edge_attributes(G,'diff_'+self.name+'_'+otherNet.name,eDiff)
        nx.set_edge_attributes(G,'weight',eDiff)

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

    ##### Saving File Functions #####
    '''
        These are functions for saving the graphs as files.
    '''

    def saveGEXFFile(self, filename, graph = None):
        if graph == None:
            nx.write_gexf(self.G, filename)
        else:
            nx.write_gexf(graph,filename)
        print('Saved gexf file %s.' % filename)

    def saveVariableHistogramPdf(self, pdfFilename):
        # save distributions in pdf format
        pp = PdfPages(pdfFilename)
        descMap = nx.get_node_attributes(self.G,'description')
        for v in self.data.columns.values.tolist():
            print(v, len(np.unique(self.data[v])))
            plt.hist(self.data.loc[np.logical_not(np.isnan(self.data[v])),v])
            plt.title(str(v))
            plt.savefig(pp,format='pdf')
            plt.cla()
        pp.close()
    
    def saveAsPickle(self, filename):
        with open(filename, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        print('Saved CorrNet as pickle file.')


    def loadFromPickle(self, filename):
        # restores itself from pickle file
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
        
        self.__dict__.update(data) 

        return


def graphFromMat(mat, nodes, attrName, nanVal=0.005):
    G = nx.Graph()

    for n in nodes:
        G.add_node(n)

    for row in range(1,len(nodes)):
        for col in range(row-1):
            edgeVal = mat[row,col]
            if np.isnan(edgeVal):
                edgeVal = nanVal
            if edgeVal == 0.0:
                edgeVal = nanVal
            G.add_edge(nodes[row],nodes[col],{attrName:float(edgeVal)})
    
    return G


def dataTableFromGraphPartitions(GList, attrName):
    '''Will extract node information from graph and convert it to a dataframe where nodes are indices.'''

    n = 10

    # get top nodes and column names
    colNames = []
    topNodes = set()
    nodeData = {}
    for G in GList:
        colNames.append(G.graph['name'])
        
        # get top n variables from this graph
        nodeData[G.graph['name']] = pd.Series(nx.get_node_attributes(G,attrName))
        nodeData[G.graph['name']].sort_values(axis=0,ascending=True, na_position='last',inplace=True)
        ind = nodeData[G.graph['name']].index.values.tolist()
        topNodes = topNodes.union(ind[0:(n-1)])
    
    # create dataframe
    topNodes = list(topNodes)
    df = pd.DataFrame(nodeData,index=nodeData[colNames[0]].index).loc[topNodes,:]

    # add variance column
    df['std'] = pd.Series(np.std(df,axis=1),index=df.index)

    # sort by variance column
    df.sort_values('std',axis=0,inplace=True,ascending=False)

    return df

