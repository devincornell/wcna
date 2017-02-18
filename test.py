
import networkx as nx
if __name__ == "__main__":
    import CorrNet
    import pandas as pd
    import numpy as np

    # create synthetic dataset
    d = dict()
    d['t'] = np.arange(100,step=0.1)
    d['x'] = np.cos(d['t'])
    d['y'] = np.cos(d['t']-0.6)
    d['z'] = np.cos(d['t']-1.3)
    d['w'] = np.cos(d['t']-2.8)
    d['b'] = d['w'] > 0
    df = pd.DataFrame(d)

    # build correlation network
    cn = CorrNet.CorrNet(df)

    # add properties
    ec = nx.betweenness_centrality(cn, weight='ar')
    nx.set_node_attributes(cn,'centrality', ec)

    # show properties
    CorrNet.edge_hist(cn,'r')

    # save
    nx.write_gexf(cn,'corrnet.gexf')
