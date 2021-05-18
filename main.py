import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import node
import network

import gen_matrix as gm
from datetime import datetime


COMPANY_NAMES = ['DBR', 'AMROBK', 'BACR-Bank', 'BNP', 'BYLAN', 'CMZB', 'CSGAG', 'DB',
                 'DZBK', 'ERGBA', 'HSBC', 'HSBC-HSBCBank', 'INTNED', 'LBW', 'NDB',
                 'SANPAO', 'SANTNDR', 'SEB', 'SOCGEN', 'UBS', 'UCBAG', 'BACF-BankNA',
                 'C', 'CRDSUI-USAInc', 'GS', 'JPM', 'MWD', 'RY', 'MIZUHBA', 'NOMURA']


def init_weights(company_names, CDS_type, detrended, moving_beta):
    """
    Initialize weight matrix
    """

    data = pd.read_csv('Data/cleaned_spreads.csv')
    if not detrended:
        # First transform nominal values to daily log returns
        log_ret = np.log(data[company_names].shift(1)/data[company_names])
        log_ret = log_ret.drop(index=[0])
    else:
        if not moving_beta:
            log_ret = pd.read_csv('Data/cleaned_spreads_detrended.csv')
        else:
            log_ret = pd.read_csv('Data/cleaned_spreads_detrended_movingbeta.csv')

    # Construct weekly W matrices
    W = []
    times = []
    for week_end in np.arange(3+21*5,len(log_ret)-2,step=5): # Skip first 3 days and last 2 days (not full weeks)
        weekly_log_ret = log_ret.iloc[week_end-21*5:week_end]
        # print(weekly_log_ret)
        times += [data['Date'].iloc[week_end]]
        if CDS_type == 'pearson':
            W += [gm.pearson_r_matrix(weekly_log_ret, company_names)]
        elif CDS_type == 'pearson_timelag':
            W += [gm.pearson_timelag_matrix(weekly_log_ret, company_names)]
        elif CDS_type == 'granger':
            W += [gm.granger_casuality(weekly_log_ret, company_names)]
    
    save_W(W, times, company_names, CDS_type, detrended, moving_beta)

    return W, times


def read_weights(company_names, CDS_type, detrended, moving_beta):
    """
    Reads weights from csv
    """
    df_W = []

    if not detrended:
        df_W = pd.read_csv(f'W_timeseries/W_{CDS_type}.csv', index_col=0)
    else:
        if not moving_beta:
            df_W = pd.read_csv(f'W_timeseries/W_{CDS_type}_detrended.csv', index_col=0)
        else:
            df_W = pd.read_csv(f'W_timeseries/W_{CDS_type}_detrended_movingbeta.csv', index_col=0)

    W = []
    for t,row in df_W.iterrows():
        # Reshape, skip date 
        W += [np.array(row[1:].values.tolist()).reshape(len(company_names), len(company_names))]

    times = df_W['Date']

    return W, times


def save_W(W, times, company_names, name, detrended, moving_beta):
    """
    Converts W matrix entries to time series, saves to csv
    """

    # Convert different W matrices to different time series
    W_to_timeseries = {'Date': {t:times[t] for t in range(len(W))}}
    for j,company_1 in enumerate(company_names):
        for i,company_2 in enumerate(company_names):
            label = company_1+'_to_'+company_2
            W_to_timeseries[label] = {t:W[t][j,i] for t in range(len(W))}

    # Convert to pandas
    df_W = pd.DataFrame(W_to_timeseries)
    if not detrended:
        df_W.to_csv(f'W_timeseries/W_{name}.csv')
    else:
        if not moving_beta: 
            df_W.to_csv(f'W_timeseries/W_{name}_detrended.csv')
        else:
            df_W.to_csv(f'W_timeseries/W_{name}_detrended_movingbeta.csv')


def init_network(w, company_names, target_bank):
    """
    Generate network and node objects based on W matrix
    """

    network_obj = network.Network(w)
    no_companies = w.shape[0]


    # First generate all nodes
    for j,company_name in enumerate(company_names):

        psi = 0
        if company_name == target_bank:
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

    return network_obj


def compute_R(network_obj, T):
    """
    Runs contagion simulation
    """
    for t in range(1,T+1):
        # First compute the h value for time point t for each node
        for node_obj in network_obj.nodes.values():
            # if node_obj.s == 'I':
            #     continue
            
            # if node_obj.s == 'U':
            node_obj.compute_h(t)

            # Secondly, update the s status
            node_obj.check_s()
        
    # At the end of the simulation, compute the Group Debtrank score
    R = network_obj.compute_R()

    return R


def run_simulation(company_names, CDS_type, W, times, detrended, moving_beta):
    """
    Runs simulation for all values of W and saves to csv
    """
    R_scores = {bank:{} for bank in COMPANY_NAMES}
    R_scores['Date'] = {t: times[t] for t in range(len(W))}

    for target_bank in COMPANY_NAMES:
        for t, w in enumerate(W):
            # Load weights into objects
            network_obj = init_network(w, COMPANY_NAMES, target_bank)
            R = compute_R(network_obj, T)
            R_scores[target_bank][t] = R
    df_R = pd.DataFrame(R_scores)

    if not detrended:
        df_R.to_csv(f'R_scores/R_score_{CDS_type}.csv')
    else:
        if not moving_beta:
            df_R.to_csv(f'R_scores/R_score_{CDS_type}_detrended.csv')
        else:
            df_R.to_csv(f'R_scores/R_score_{CDS_type}_detrended_movingbeta.csv')


if __name__ == '__main__':

    # Parameter values for the simulation
    T = 2

    # Select method:
    # pearson = True
    # pearson_timelag = False
    # Granger_caus = False

    CDS_types = ['pearson', 'pearson_timelag', 'granger']
    # CDS_types = ['pearson', 'pearson_timelag']
    # CDS_types = ['pearson', 'granger']
    # CDS_types = ['pearson_timelag', 'granger']
    # CDS_types = ['granger']
    # CDS_types = ['pearson']
    # CDS_types = ['pearson_timelag']

    # Generate weights or read weights:
    gen_weights = False
    detrended = True
    moving_beta = False

    for CDS_type in CDS_types:
        
        print(50*'-')
        start_time = datetime.now()
        if gen_weights:
            print(f'Generating weights for {CDS_type} method')
            W, times = init_weights(COMPANY_NAMES, CDS_type, detrended, moving_beta)
        else:
            print(f'Reading in weights for {CDS_type} method')
            W, times = read_weights(COMPANY_NAMES, CDS_type, detrended, moving_beta)

        print('Running simulation...')
        run_simulation(COMPANY_NAMES, CDS_type, W, times, detrended, moving_beta)
        print(f'Simulation Finished, runtime={datetime.now()-start_time}.')
        print(50*'-')