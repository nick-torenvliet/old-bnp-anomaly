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
#
# *DiagGauss* observation model, without moves
# --------------------------------------------
#
# Start with too many clusters (K=25)

gamma = 5.0
sF = 5.0
K = 25

diag1_trained_model, diag1_info_dict = bnpy.run(
    init_data, 'DPMixtureModel', 'DiagGauss', 'memoVB',
    output_path=('/tmp/faithful/' + 
        'trymoves-K=%d-gamma=%s-lik=DiagGauss-ECovMat=%s*eye-moves=none/' % (
            K, gamma, sF)),
    nLap=2, nTask=1, nBatch=p, convergeThr=0.0001,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamplesbydist',
    )
