import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import scipy.stats as stats


def pearson_r_matrix(data, company_names):
    """
    Constructs a pearson r correlation matrix, after which a weights matrix is constructed
    """
    N = len(company_names)
    pearson_matrix = np.zeros((N,N))

    # Fill in correlation matrix
    for j in range(N):
        for i in range(j,N):
            pearson_r = stats.pearsonr(data[company_names[j]], data[company_names[i]])

            # Because of correlation is the same both ways, it only needs to be computed once,
            # after which it can be filled in at both [j,i] and [i,j]
            if pearson_r[1] < 0.05:  #Check for significance
                pearson_matrix[j,i] = pearson_r[0]
                pearson_matrix[i,j] = pearson_r[0]

    # Compute weights
    W = np.zeros((N,N))
    for j in range(N):
        for i in range(j,N):

            # Skip if correlation is zero
            if pearson_matrix[j,i] == 0:
                continue

            w_ji = 1 - np.sqrt(2*(1-pearson_matrix[j,i]))/2
            W[j,i] = w_ji
            W[i,j] = w_ji

    return W

def pearson_timelag_matrix(data, company_names):
    """
    Constructs a pearson r correlation matrix using 5 time lags, after which a weights matrix is constructed
    """
    N = len(company_names)
    pearson_matrix = np.zeros((N,N))

    # Fill in correlation matrix
    for j in range(N):
        for i in range(N):

            # Compute correlation for 5 time lags:
            all_corr = []
            for lag in range(1,6):
                pearson_r = stats.pearsonr(data[company_names[j]][:-lag], data[company_names[i]].shift(-lag)[:-lag])
                if pearson_r[1] < 0.05:  #Check for significance
                    all_corr += [pearson_r[0]]

            if len(all_corr) > 0:
                pearson_matrix[j,i] = np.average(all_corr)

    # Compute weights
    W = np.zeros((N,N))
    for j in range(N):
        for i in range(N):

            # Skip if correlation is zero
            if pearson_matrix[j,i] == 0:
                continue

            w_ji = 1 - np.sqrt(2*(1-pearson_matrix[j,i]))/2
            W[j,i] = w_ji

    return W


def granger_casuality(data, company_names):
    """"
    Constructs a matrix with the wights for the Granger causality method
    """
    N = len(company_names)
    granger_weight_matrix = np.zeros((N,N))
    
    # Fill in the correlation matrix
    for j in range(N):
        for i in range(N):
            # Skip if same company
            if j==i:
                granger_weight_matrix[j,i] = 1
            else:
                ar = np.stack((data.iloc[:-1,j],data.iloc[1:,i]), axis=1)
                # print(ar)
                granger_test = grangercausalitytests(ar, 1, verbose=False)
                # print(granger_test)
                p_val = granger_test[1][0]['ssr_ftest'][1]
                if p_val < 0.05:
                    R_sq = granger_test[1][1][1].rsquared
                    # print(R_sq)
                    granger_weight_matrix[j,i] = R_sq
            
    return granger_weight_matrix
