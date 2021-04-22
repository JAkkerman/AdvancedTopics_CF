import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import node
import network

import gen_matrix as gm


COMPANY_NAMES = ['DBR', 'AMROBK', 'BACR-Bank', 'BNP', 'BYLAN', 'CMZB', 'CSGAG', 'DB',
                 'DZBK', 'ERGBA', 'HSBC', 'HSBC-HSBCBank', 'INTNED', 'LBW', 'NDB',
                 'SANPAO', 'SANTNDR', 'SEB', 'SOCGEN', 'UBS', 'UCBAG', 'BACF-BankNA',
                 'C', 'CRDSUI-USAInc', 'GS', 'JPM', 'MWD', 'RY', 'MIZUHBA', 'NOMURA']


def init_weights(company_names, pearson, Granger_caus):
    """
    Initialize weight matrix
    """
    data = pd.read_csv('cleaned_spreads.csv')

    # First transform nominal values to daily log returns
    log_ret = np.log(data[company_names].shift(1)/data[company_names])
    log_ret = log_ret.drop(index=[0])

    # Construct weekly W matrices
    W = []
    for week_end in np.arange(3+21*5,len(log_ret)-2,step=5): # Skip first 3 days and last 2 days (not full weeks)
        # print(data.iloc[weekstart:weekstart+5])
        weekly_log_ret = log_ret.iloc[week_end-21*5:week_end]
        if pearson:
            W += [gm.pearson_r_matrix(weekly_log_ret, company_names)]
        elif Granger_caus:
            W += [gm.granger_casuality(weekly_log_ret, company_names)]
    
    if pearson:
        save_W(W, company_names, 'pearson')
    elif Granger_caus:
        save_W(W, company_names, 'Granger_Caus')

    # plt.imshow(W[-2])
    # plt.colorbar()
    # plt.show()

    # w_over_time = [W[i][1,12] for i in range(len(W))]
    # plt.plot(range(len(W)), w_over_time)
    # plt.title('w for ABN AMRO and ING Bank')
    # plt.show()

    return W


def read_weights(company_names, pearson, Granger_caus):
    """
    Reads weights from csv
    """
    df_W = []

    if pearson:
        df_W = pd.read_csv('W_timeseries/W_pearson.csv', index_col=0)
    # print(df_W.head())

    W = {}
    for t,row in df_W.iterrows():
        Wt = np.array(row.values.tolist()).reshape(len(company_names), len(company_names))
        W[t] = Wt

    return W


def save_W(W, company_names, name):
    """
    Converts W matrix entries to time series, saves to csv
    """

    # Convert different W matrices to different time series
    W_to_timeseries = {}
    for j,company_1 in enumerate(company_names):
        for i,company_2 in enumerate(company_names):
            label = company_1+'_to_'+company_2
            W_to_timeseries[label] = {t:W[t][j,i] for t in range(len(W))}

    # Convert to pandas
    df_W = pd.DataFrame(W_to_timeseries)
    df_W.to_csv(f'W_timeseries/W_{name}.csv')


def init_network(w, company_names):
    """
    Generate network and node objects based on W matrix
    """

    network_obj = network.Network(w)
    no_companies = w.shape[0]


    # First generate all nodes
    for j,company_name in enumerate(company_names):

        # TODO: Let psi vary for different banks
        psi = 0
        if company_name == 'AMROBK':
            psi = 1

        node_obj = node.Node(network_obj, company_name, j, psi)
        node_obj.check_s()
        network_obj.nodes[company_name] = node_obj

    # Read out the connections and add to node objects
    for j in range(no_companies):
        company_1  = company_names[j]
        for i in range(no_companies):
            if i != j:
                company_2 = company_names[i]
                network_obj.nodes[company_1].connections[company_2] = (network_obj.nodes[company_2], w[j,i])


    # for node_obj in network_obj.nodes.values():
    #     print(node_obj.connections)

    return network_obj


def run_simulation(network_obj, T):
    """
    Runs contagion simulation
    """
    for t in range(T):
        # First compute the h value for time point t for each node
        for node_obj in network_obj.nodes.values():
            node_obj.compute_h(t)

            # Secondly, update the s status
            node_obj.check_s()
        
    # At the end of the simulation, compute the Group Debtrank score
    R = network_obj.compute_R()

    return R


if __name__ == '__main__':

    # Parameter values for the simulation
    T = 3

    # Select method:
    pearson = False
    Granger_caus = True

    # Generate weights or read weights:
    gen_weights = True

    if gen_weights:
        W = init_weights(COMPANY_NAMES, pearson, Granger_caus)
    else:
        W = read_weights(COMPANY_NAMES, pearson, Granger_caus)

    # TODO now only for first matrix, later for all matrices
    # for w in [W[0]]:
    #     # Load weights into objects
    #     network_obj = init_network(w, COMPANY_NAMES)
    #     R = run_simulation(network_obj, T)
    #     print('R score for ABN AMRO at t=0: ', R)