import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import node
import network

import gen_matrix as gm

def init_weights(pearson=True):
    """
    Initialize weight matrix
    """
    data = pd.read_csv('cleaned_spreads.csv')

    company_names = data.columns[1:]

    # First transform nominal values to daily log returns
    log_ret = np.log(data[company_names].shift(1)/data[company_names])
    # log_ret = log_ret.drop(index=[0]).iloc[:700]
    log_ret = log_ret.drop(index=[0])
    # print(log_ret)

    # Construct weekly W matrices
    W = []
    for week_end in np.arange(3+21*5,len(log_ret)-2,step=5): # Skip first 3 days and last 2 days (not full weeks)
        # print(data.iloc[weekstart:weekstart+5])
        weekly_log_ret = log_ret.iloc[week_end-21*5:week_end]
        if pearson:
            W += [gm.pearson_r_matrix(weekly_log_ret)]

    # print(W)

    plt.imshow(W[-2])
    plt.colorbar()
    plt.show()

    w_over_time = [W[i][1,12] for i in range(len(W))]
    plt.plot(range(len(W)), w_over_time)
    plt.title('w for ABN AMRO and ING Bank')
    plt.show()

    return data, log_ret, W, company_names

def init_network(W, company_names):
    """
    Generate network and node objects based on W matrix
    """
    # company_names = data.columns[1:]

    network_obj = network.Network(W)
    for j,company_name in enumerate(company_names):
        connections = {company_names[i]: W[j,i] for i in range(len(company_names)) if company_names[i] != company_name}
        node_obj = node.Node(company_name, j, connections)
        network_obj.nodes += [node_obj]

    # print(network_obj.nodes)


if __name__ == '__main__':

    data, log_ret, W, company_names = init_weights()
    # network = init_network(W, company_names)