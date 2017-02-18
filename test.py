
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

    df = pd.DataFrame(d)

    cn = CorrNet.CorrNet(df)

    ec = nx.betweenness_centrality(cn, weight='ar')
    nx.set_node_attributes(cn,'centrality', ec)

    nx.write_gexf(cn,'corrnet.gexf')
