import numpy as np
import pandas as pd

import scipy.stats as stats

def pearson_r_matrix(data):
    """
    Constructs a pearson r correlation matrix, after which a weights matrix is constructed
    """
    company_names = data.columns[1:]
    N = len(company_names)
    pearson_matrix = np.zeros((N,N))

    # Fill in correlation matrix
    for j in range(N):
        for i in range(j,N):
            pearson_r = stats.pearsonr(data[company_names[j]], data[company_names[i]])
            # print(pearson_r)

            # Because of correlation is the same both ways, it only needs to be computed once,
            # after which it can be filled in at both [j,i] and [i,j]
            if pearson_r[1] < 0.05:  #Check for significance
                pearson_matrix[j,i] = pearson_r[0]
                pearson_matrix[i,j] = pearson_r[0]
            else:
                pearson_matrix[j,i] = np.nan
                pearson_matrix[i,j] = np.nan

    # Compute weights
    W = np.zeros((N,N))
    for j in range(N):
        for i in range(j,N):
            w_ji = 1 - np.sqrt(2*(1-pearson_matrix[j,i]))/2
            # w_ji = pearson_matrix[j,i]
            W[j,i] = w_ji
            W[i,j] = w_ji

    return W