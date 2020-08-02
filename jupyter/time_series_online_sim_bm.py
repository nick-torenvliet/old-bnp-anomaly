import bnpy
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up online streaming context
n = 5000
b = 10
p = int(n/b)
assert (p>0)
synthetic = {'Unnamed: 0': list(range(0,5000)),
        'anomaly': list(range(0,5000))
        }
some_data = pd.DataFrame(synthetic, columns = ['Unnamed: 0', 'anomaly'])
print (some_data.head())

# path = '/home/torenvln/git/bnp-anomaly/data/anomaly0245.csv'
# all_data = pd.read_csv(path) 
# some_data = all_data.head(5000)
# print(all_data.head(5))
init_data = bnpy.data.XData.from_dataframe(some_data)

###############################################################################
# Setup: Determine specific settings of the proposals
# ---------------------------------------------------

merge_kwargs = dict(
    m_startLap=10,
    m_pair_ranking_procedure='total_size',
    m_pair_ranking_direction='descending',
    )

delete_kwargs = dict(
    d_startLap=20,
    d_nRefineSteps=50,
    )

###############################################################################
#
# *Gauss* observation model
# -------------------------
#
# Start with too many clusters (K=25)
# Use merges and deletes to reduce to a better set.

gamma = 5.0
sF = 5.0
K = 25

full_trained_model, full_info_dict = bnpy.run(
    init_data, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path=('/tmp/faithful/' + 
        'trymoves-K=%d-gamma=%s-lik-Gauss-ECovMat=%s*eye-moves=merge,delete,shuffle/' % (
            K, gamma, sF)),
    nLap=100, nTask=1, nBatch=1,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamplesbydist',
    moves='merge,delete,shuffle',
    **dict(list(delete_kwargs.items()) + list(merge_kwargs.items())))


